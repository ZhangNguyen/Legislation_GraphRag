from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.rag.document_header import DocumentHeader, extract_document_header

ARTICLE_RE = re.compile(r"^Điều\s+(\d+)(?:[\.:\-\)]?\s*(.*))?$", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+(.+)$")
POINT_RE = re.compile(r"^([a-zđ])\)\s+(.+)$", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\u2022]\s+(.+)$")
UPPER_SECTION_RE = re.compile(r"^(PHẦN|CHƯƠNG|MỤC|TIỂU MỤC)\b", re.IGNORECASE)


@dataclass
class LegalNode:
    node_id: str
    doc_id: str
    node_type: str
    label: str
    text: str
    parent_id: Optional[str]
    order_index: int
    level: int
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    item: Optional[str] = None
    path_title: str = ""
    children_ids: List[str] = field(default_factory=list)


@dataclass
class ParsedLegalDocument:
    header: DocumentHeader
    nodes: List[LegalNode]


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "node"


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").splitlines():
        line = raw.replace("\ufeff", " ").strip()
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


def _new_node_id(doc_id: str, node_type: str, *parts: object) -> str:
    body = "__".join(_slugify(str(x)) for x in parts if x not in (None, ""))
    return f"{doc_id}::{node_type}::{body}" if body else f"{doc_id}::{node_type}"


def _build_path(node: LegalNode, node_index: Dict[str, LegalNode]) -> str:
    parts: List[str] = []
    cur: Optional[LegalNode] = node
    while cur is not None:
        if cur.node_type != "document":
            parts.append(_norm_space(cur.label))
        cur = node_index.get(cur.parent_id) if cur.parent_id else None
    parts.reverse()
    return " > ".join(p for p in parts if p)


def _first_line(text: str) -> str:
    for line in str(text or "").splitlines():
        line = _norm_space(line)
        if line:
            return line
    return ""


def parse_legal_document(text: str, *, fallback_doc_name: str = "") -> ParsedLegalDocument:
    header = extract_document_header(text, fallback_name=fallback_doc_name)
    all_lines = _clean_lines(text)
    body_lines = all_lines[header.body_start_index:] if header.body_start_index < len(all_lines) else all_lines

    doc_title = header.official_title or header.file_stem
    doc_node = LegalNode(
        node_id=f"{header.doc_id}::document",
        doc_id=header.doc_id,
        node_type="document",
        label=doc_title,
        text=_norm_space("\n".join(x for x in [header.title_block, header.lead_block] if x)),
        parent_id=None,
        order_index=0,
        level=0,
    )

    nodes: List[LegalNode] = [doc_node]
    node_index: Dict[str, LegalNode] = {doc_node.node_id: doc_node}

    current_article: Optional[LegalNode] = None
    current_clause: Optional[LegalNode] = None
    current_point: Optional[LegalNode] = None
    current_item: Optional[LegalNode] = None
    current_section: Optional[LegalNode] = None
    current_anchor: LegalNode = doc_node
    order = 1

    for line in body_lines:
        article_m = ARTICLE_RE.match(line)
        if article_m:
            article_num = article_m.group(1)
            article_suffix = _norm_space(article_m.group(2) or "")
            article = f"Điều {article_num}"
            label = article if not article_suffix else f"{article}. {article_suffix}"
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "article", article_num),
                doc_id=header.doc_id,
                node_type="article",
                label=label,
                text=line,
                parent_id=doc_node.node_id,
                order_index=order,
                level=1,
                article=article,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            doc_node.children_ids.append(node.node_id)
            current_article = node
            current_clause = None
            current_point = None
            current_item = None
            current_section = None
            current_anchor = node
            continue

        if UPPER_SECTION_RE.match(line):
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "section", line),
                doc_id=header.doc_id,
                node_type="section",
                label=line,
                text=line,
                parent_id=doc_node.node_id,
                order_index=order,
                level=1,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            doc_node.children_ids.append(node.node_id)
            current_section = node
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_anchor = node
            continue

        clause_m = NUMBERED_ITEM_RE.match(line)
        if clause_m and current_article is not None:
            clause_num = clause_m.group(1)
            clause = f"Khoản {clause_num}"
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "clause", current_article.article or current_article.label, clause_num),
                doc_id=header.doc_id,
                node_type="clause",
                label=clause,
                text=line,
                parent_id=current_article.node_id,
                order_index=order,
                level=2,
                article=current_article.article,
                clause=clause,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            current_article.children_ids.append(node.node_id)
            current_clause = node
            current_point = None
            current_item = None
            current_anchor = node
            continue

        point_m = POINT_RE.match(line)
        if point_m and current_clause is not None:
            ch = point_m.group(1).lower()
            point = f"Điểm {ch}"
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "point", current_article.article or "", current_clause.clause or "", ch),
                doc_id=header.doc_id,
                node_type="point",
                label=point,
                text=line,
                parent_id=current_clause.node_id,
                order_index=order,
                level=3,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=point,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            current_clause.children_ids.append(node.node_id)
            current_point = node
            current_item = None
            current_anchor = node
            continue

        bullet_m = BULLET_RE.match(line)
        if bullet_m:
            parent = current_point or current_clause or current_item or current_article or current_section or doc_node
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "bullet", parent.node_id, order),
                doc_id=header.doc_id,
                node_type="bullet",
                label="Bullet",
                text=line,
                parent_id=parent.node_id,
                order_index=order,
                level=parent.level + 1,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=current_point.point if current_point else None,
                item=current_item.item if current_item else None,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            parent.children_ids.append(node.node_id)
            current_anchor = node
            continue

        current_anchor.text = _norm_space(f"{current_anchor.text}\n{line}")

    for node in nodes:
        node.path_title = _build_path(node, node_index)

    return ParsedLegalDocument(header=header, nodes=nodes)


def _descendant_ids(node_id: str, node_index: Dict[str, LegalNode]) -> List[str]:
    out: List[LegalNode] = []
    queue = [node_id]
    while queue:
        cur_id = queue.pop(0)
        cur = node_index.get(cur_id)
        if not cur:
            continue
        for child_id in cur.children_ids:
            child = node_index.get(child_id)
            if not child:
                continue
            out.append(child)
            queue.append(child_id)
    out.sort(key=lambda n: n.order_index)
    return [n.node_id for n in out]


def _doc_sketch_text(header: DocumentHeader, article_labels: List[str]) -> str:
    parts: List[str] = []
    if header.doc_type:
        parts.append(f"Loại văn bản: {header.doc_type}")
    if header.official_title or header.file_stem:
        parts.append(f"Tiêu đề: {header.official_title or header.file_stem}")
    if header.doc_number:
        parts.append(f"Số văn bản: {header.doc_number}")
    if header.issuing_agency:
        parts.append(f"Cơ quan ban hành: {header.issuing_agency}")
    if header.date_raw:
        parts.append(f"Ngày ban hành: {header.date_raw}")
    if article_labels:
        preview = article_labels[:15]
        parts.append("Các điều chính:\n" + "\n".join(f"- {x}" for x in preview))
    return "\n".join(parts).strip()


def _provision_retrieval_text(header: DocumentHeader, node: LegalNode) -> str:
    parts = [
        f"Văn bản: {header.official_title or header.file_stem}",
        f"Vị trí: {node.path_title or node.label}",
        f"Nội dung: {node.text}",
    ]
    return "\n".join(_norm_space(x) for x in parts if _norm_space(x)).strip()


def _provision_rerank_text(header: DocumentHeader, node: LegalNode, *, parent_heading: str = "") -> str:
    parts = [
        f"Văn bản: {header.official_title or header.file_stem}",
        f"Vị trí pháp lý: {node.path_title or node.label}",
    ]
    if parent_heading and parent_heading != node.label:
        parts.append(f"Neo cha: {parent_heading}")
    parts.append(f"Chính node: {node.text}")
    return "\n".join(_norm_space(x) for x in parts if _norm_space(x)).strip()


def _truncate(text: str, limit: int) -> str:
    clean = _norm_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def build_chunks(parsed: ParsedLegalDocument) -> List[dict]:
    header = parsed.header
    nodes = parsed.nodes
    node_index = {node.node_id: node for node in nodes}
    chunks: List[dict] = []
    article_nodes: List[LegalNode] = [n for n in nodes if n.node_type == "article"]

    for node in nodes:
        if node.node_type == "document":
            continue
        parent = node_index.get(node.parent_id) if node.parent_id else None
        sibling_ids = parent.children_ids if parent else []
        sibling_ids = [x for x in sibling_ids if x != node.node_id]
        heading_title = _first_line(node.text) or node.label
        metadata = {
            "artifact_type": "evidence",
            "doc_id": header.doc_id,
            "node_id": node.node_id,
            "chunk_id": node.node_id,
            "node_type": node.node_type,
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids),
            "sibling_ids": sibling_ids[:12],
            "order_index": node.order_index,
            "level": node.level,
            "path_title": node.path_title,
            "heading_title": heading_title,
            "title": heading_title,
            "article": node.article,
            "clause": node.clause,
            "point": node.point,
            "item": node.item,
            "doc_type": header.doc_type or "Unknown",
            "official_title": header.official_title or header.file_stem,
            "issuing_agency": header.issuing_agency,
            "doc_number": header.doc_number,
            "date_raw": header.date_raw,
            "year": header.year or 0,
            "law_name": header.official_title or header.file_stem,
            "law_type": header.doc_type or "Unknown",
            "source": header.issuing_agency or "LocalFile",
            "file_stem": header.file_stem,
        }
        retrieval_text = _provision_retrieval_text(header, node)
        rerank_text = _provision_rerank_text(header, node, parent_heading=parent.label if parent else "")
        chunks.append(
            {
                "text": _norm_space(node.text),
                "snippet": _truncate(node.text, 700),
                "retrieval_text": retrieval_text,
                "rerank_text": rerank_text,
                "metadata": metadata,
            }
        )

    article_bundle_ids: List[str] = []
    for article in article_nodes:
        descendant_ids = _descendant_ids(article.node_id, node_index)
        source_ids = [article.node_id] + descendant_ids
        source_nodes = [node_index[nid] for nid in source_ids if nid in node_index]
        bundle_lines = [_norm_space(n.text) for n in source_nodes if _norm_space(n.text)]
        bundle_text = "\n".join(bundle_lines).strip()
        if not bundle_text:
            continue
        bundle_id = f"{article.node_id}::bundle"
        article_bundle_ids.append(bundle_id)
        metadata = {
            "artifact_type": "article_bundle",
            "doc_id": header.doc_id,
            "node_id": bundle_id,
            "chunk_id": bundle_id,
            "node_type": "article_bundle",
            "title": article.label,
            "heading_title": article.label,
            "path_title": article.label,
            "article": article.article,
            "clause": None,
            "point": None,
            "parent_id": None,
            "children_ids": [],
            "sibling_ids": [],
            "source_node_ids": source_ids,
            "doc_type": header.doc_type or "Unknown",
            "official_title": header.official_title or header.file_stem,
            "issuing_agency": header.issuing_agency,
            "doc_number": header.doc_number,
            "date_raw": header.date_raw,
            "year": header.year or 0,
            "law_name": header.official_title or header.file_stem,
            "law_type": header.doc_type or "Unknown",
            "source": header.issuing_agency or "LocalFile",
            "file_stem": header.file_stem,
            "order_index": article.order_index,
            "level": 1,
        }
        retrieval_text = "\n".join(
            [
                f"Văn bản: {header.official_title or header.file_stem}",
                f"Điều: {article.label}",
                f"Nội dung điều: {bundle_text}",
            ]
        ).strip()
        rerank_text = "\n".join(
            [
                f"Văn bản: {header.official_title or header.file_stem}",
                f"Vị trí pháp lý: {article.label}",
                f"Nội dung điều: {_truncate(bundle_text, 1800)}",
            ]
        ).strip()
        chunks.append(
            {
                "text": bundle_text,
                "snippet": _truncate(bundle_text, 700),
                "retrieval_text": retrieval_text,
                "rerank_text": rerank_text,
                "metadata": metadata,
            }
        )

    doc_sketch_id = f"{header.doc_id}::doc_sketch"
    article_labels = [article.label for article in article_nodes]
    doc_sketch_text = _doc_sketch_text(header, article_labels)
    if doc_sketch_text:
        metadata = {
            "artifact_type": "doc_sketch",
            "doc_id": header.doc_id,
            "node_id": doc_sketch_id,
            "chunk_id": doc_sketch_id,
            "node_type": "doc_sketch",
            "title": header.official_title or header.file_stem,
            "heading_title": header.official_title or header.file_stem,
            "path_title": "Tóm tắt cấu trúc văn bản",
            "article": None,
            "clause": None,
            "point": None,
            "parent_id": None,
            "children_ids": [],
            "sibling_ids": [],
            "source_node_ids": article_bundle_ids,
            "doc_type": header.doc_type or "Unknown",
            "official_title": header.official_title or header.file_stem,
            "issuing_agency": header.issuing_agency,
            "doc_number": header.doc_number,
            "date_raw": header.date_raw,
            "year": header.year or 0,
            "law_name": header.official_title or header.file_stem,
            "law_type": header.doc_type or "Unknown",
            "source": header.issuing_agency or "LocalFile",
            "file_stem": header.file_stem,
            "order_index": 0,
            "level": 0,
        }
        chunks.append(
            {
                "text": doc_sketch_text,
                "snippet": _truncate(doc_sketch_text, 700),
                "retrieval_text": doc_sketch_text,
                "rerank_text": _truncate(doc_sketch_text, 1800),
                "metadata": metadata,
            }
        )

    return chunks


def parse_legal_text(text: str, fallback_doc_name: str = "") -> List[LegalNode]:
    return parse_legal_document(text, fallback_doc_name=fallback_doc_name).nodes


def legal_chunk(text: str, max_chars: int = 1600, overlap: int = 120, fallback_doc_name: str = "") -> List[dict]:
    _ = (max_chars, overlap)
    parsed = parse_legal_document(text, fallback_doc_name=fallback_doc_name)
    return build_chunks(parsed)
