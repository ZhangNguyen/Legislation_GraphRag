from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.rag.document_header import DocumentHeader, extract_document_header

ARTICLE_RE = re.compile(r"^Điều\s+(\d+)([\.:\-\)]?\s*.*)?$", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+(.+)$")
POINT_RE = re.compile(r"^([a-zđ])\)\s+(.+)$", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\u2022]\s+(.+)$")
UPPER_SECTION_RE = re.compile(r"^(PHẦN|CHƯƠNG|MỤC|TIỂU MỤC)\b", re.IGNORECASE)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "node"


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
    if body:
        return f"{doc_id}::{node_type}::{body}"
    return f"{doc_id}::{node_type}"


def _node_heading(node: LegalNode) -> str:
    return _norm_space(node.label)


def _build_path(node: LegalNode, node_index: Dict[str, LegalNode]) -> str:
    parts: List[str] = []
    cur: Optional[LegalNode] = node
    while cur is not None:
        if cur.node_type != "document":
            parts.append(_node_heading(cur))
        cur = node_index.get(cur.parent_id) if cur.parent_id else None
    parts.reverse()
    return " > ".join(p for p in parts if p)


def parse_legal_document(text: str, *, fallback_doc_name: str = "") -> ParsedLegalDocument:
    header = extract_document_header(text, fallback_name=fallback_doc_name)
    all_lines = _clean_lines(text)
    body_lines = all_lines[header.body_start_index:] if header.body_start_index < len(all_lines) else all_lines

    doc_node = LegalNode(
        node_id=f"{header.doc_id}::document",
        doc_id=header.doc_id,
        node_type="document",
        label=header.official_title or header.file_stem,
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
            label = f"Điều {article_num}"
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "article", article_num),
                doc_id=header.doc_id,
                node_type="article",
                label=label,
                text=line,
                parent_id=doc_node.node_id,
                order_index=order,
                level=1,
                article=label,
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

        number_m = NUMBERED_ITEM_RE.match(line)
        if number_m:
            number = number_m.group(1)
            parent = current_article or current_section or doc_node
            node_type = "clause" if current_article is not None else "item"
            clause = f"Khoản {number}" if node_type == "clause" else None
            item = f"Mục {number}" if node_type == "item" else None
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, node_type, parent.node_id, number),
                doc_id=header.doc_id,
                node_type=node_type,
                label=line,
                text=line,
                parent_id=parent.node_id,
                order_index=order,
                level=parent.level + 1,
                article=current_article.article if current_article else None,
                clause=clause,
                item=item,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            parent.children_ids.append(node.node_id)
            if node_type == "clause":
                current_clause = node
            else:
                current_item = node
            current_point = None
            current_anchor = node
            continue

        point_m = POINT_RE.match(line)
        if point_m:
            ch = point_m.group(1).lower()
            parent = current_clause or current_item or current_article or current_section or doc_node
            node = LegalNode(
                node_id=_new_node_id(header.doc_id, "point", parent.node_id, ch),
                doc_id=header.doc_id,
                node_type="point",
                label=f"Điểm {ch}",
                text=line,
                parent_id=parent.node_id,
                order_index=order,
                level=parent.level + 1,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=f"Điểm {ch}",
                item=current_item.item if current_item else None,
            )
            order += 1
            nodes.append(node)
            node_index[node.node_id] = node
            parent.children_ids.append(node.node_id)
            current_point = node
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


def _doc_summary_text(header: DocumentHeader) -> str:
    parts = []
    if header.doc_type:
        parts.append(f"Loại văn bản: {header.doc_type}")
    if header.official_title:
        parts.append(f"Tiêu đề: {header.official_title}")
    if header.issuing_agency:
        parts.append(f"Cơ quan ban hành: {header.issuing_agency}")
    if header.doc_number:
        parts.append(f"Số văn bản: {header.doc_number}")
    if header.lead_block:
        parts.append(f"Dẫn nhập: {header.lead_block}")
    return "\n".join(parts).strip()


def _build_retrieval_text(header: DocumentHeader, node: LegalNode) -> str:
    parts = [
        f"[DOC_TYPE] {header.doc_type or 'Unknown'}",
        f"[OFFICIAL_TITLE] {header.official_title or header.file_stem}",
    ]
    if header.issuing_agency:
        parts.append(f"[ISSUING_AGENCY] {header.issuing_agency}")
    if header.doc_number:
        parts.append(f"[DOC_NUMBER] {header.doc_number}")
    if header.lead_block:
        parts.append(f"[LEAD_BLOCK] {header.lead_block}")
    if node.path_title:
        parts.append(f"[PATH] {node.path_title}")
    parts.append(f"[NODE_TYPE] {node.node_type}")
    parts.append(f"[CHUNK] {node.text}")
    return "\n".join(x for x in parts if _norm_space(x)).strip()


def build_chunks(parsed: ParsedLegalDocument) -> List[dict]:
    header = parsed.header
    chunks: List[dict] = []
    node_index = {node.node_id: node for node in parsed.nodes}

    for node in parsed.nodes:
        if node.node_type == "document":
            continue
        parent = node_index.get(node.parent_id) if node.parent_id else None
        sibling_ids = parent.children_ids if parent else []
        sibling_ids = [x for x in sibling_ids if x != node.node_id]
        metadata = {
            "doc_id": header.doc_id,
            "node_id": node.node_id,
            "chunk_id": node.node_id,
            "node_type": node.node_type,
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids),
            "sibling_ids": sibling_ids[:8],
            "order_index": node.order_index,
            "level": node.level,
            "path_title": node.path_title,
            "article": node.article,
            "clause": node.clause,
            "point": node.point,
            "item": node.item,
            "doc_type": header.doc_type or "Unknown",
            "official_title": header.official_title or header.file_stem,
            "title_block": header.title_block,
            "lead_block": header.lead_block,
            "issuing_agency": header.issuing_agency,
            "doc_number": header.doc_number,
            "date_raw": header.date_raw,
            "year": header.year or 0,
            "law_name": header.official_title or header.file_stem,
            "law_type": header.doc_type or "Unknown",
            "source": header.issuing_agency or "LocalFile",
            "doc_summary": _doc_summary_text(header),
            "file_stem": header.file_stem,
        }
        retrieval_text = _build_retrieval_text(header, node)
        chunks.append(
            {
                "text": node.text,
                "snippet": node.text[:600],
                "retrieval_text": retrieval_text,
                "rerank_text": retrieval_text,
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
