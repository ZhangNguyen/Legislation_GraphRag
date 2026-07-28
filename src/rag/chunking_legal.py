from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.rag.document_header import DocumentHeader, extract_document_header

ARTICLE_LABEL = "\u0110i\u1ec1u"
CLAUSE_LABEL = "Kho\u1ea3n"
POINT_LABEL = "\u0110i\u1ec3m"

ARTICLE_RE = re.compile(r"^dieu\s+(\d+[a-z]?)(?:[\.:;\-\)]?\s*(.*))?$", re.IGNORECASE)
UPPER_SECTION_RE = re.compile(r"^(phan|chuong|muc|tieu muc)\b", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^phu luc\b", re.IGNORECASE)
ATTACHMENT_RE = re.compile(r"^(de an|ke hoach|chuong trinh)$", re.IGNORECASE)
ROMAN_SECTION_RE = re.compile(r"^(?P<label>[IVXLCDM]{1,8})[\.\)]\s+(?P<title>.+)$")
ALPHA_SECTION_RE = re.compile(r"^(?P<label>[A-Z])[\.\)]\s+(?P<title>.+)$")
DECIMAL_ITEM_RE = re.compile(r"^(?P<label>\d+(?:\.\d+){1,})\.?\s+(?P<title>.+)$")
NUMBERED_ITEM_RE = re.compile(r"^(?P<label>\d+)\.\s+(?P<title>.+)$")
NUMBER_PAREN_RE = re.compile(r"^(?P<label>\d+)\)\s+(?P<title>.+)$")
PAREN_ITEM_RE = re.compile(r"^\((?P<label>[a-z]|[ivxlcdm]{1,8}|\d+)\)\s+(?P<title>.+)$", re.IGNORECASE)
POINT_RE = re.compile(r"^(?P<label>[a-z\u0111])\)\s+(?P<title>.+)$", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\+\*\u2022]\s+(.+)$")
INLINE_DECIMAL_SPLIT_RE = re.compile(r"\s+(?=\d+(?:\.\d+){1,}\.?\s+)")

SECTION_LIKE_TYPES = {"section", "appendix", "attachment", "roman_section", "alpha_section"}
ITEM_LIKE_TYPES = {"item", "decimal_item", "list_item"}
TABLE_PARENT_TYPES = SECTION_LIKE_TYPES | {"article", "clause", "point"} | ITEM_LIKE_TYPES


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


def _norm_space(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _ascii_key(text: object) -> str:
    raw = unicodedata.normalize("NFD", _norm_space(text))
    raw = "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")
    raw = raw.replace("\u0111", "d").replace("\u0110", "D")
    return raw.lower()


def _slugify(text: object) -> str:
    raw = _ascii_key(text)
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "node"


def _looks_like_table_line(line: str) -> bool:
    if "|" not in line:
        return False
    cells = [_norm_space(x) for x in line.split("|")]
    cells = [x for x in cells if x]
    if len(cells) < 2:
        return False
    key = _ascii_key(" ".join(cells[:2]))
    if key.startswith(("so:", "so ", "cong hoa", "doc lap", "noi nhan", "luu:")):
        return False
    return True


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").splitlines():
        line = raw.replace("\ufeff", " ").strip()
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if not _looks_like_table_line(line) and NUMBERED_ITEM_RE.match(line):
            parts = [p.strip() for p in INLINE_DECIMAL_SPLIT_RE.split(line) if p.strip()]
            if len(parts) > 1 and all(i == 0 or DECIMAL_ITEM_RE.match(p) for i, p in enumerate(parts)):
                out.extend(parts)
                continue
        out.append(line)
    return out


def _new_node_id(doc_id: str, node_type: str, order: int, *parts: object) -> str:
    body_parts = [f"{order:06d}"] + [_slugify(x) for x in parts if _norm_space(x)]
    return f"{doc_id}::{node_type}::{'__'.join(body_parts)}"


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


def _is_footer_start(line: str) -> bool:
    key = _ascii_key(line).strip(" :-")
    if not key:
        return False
    if key.startswith(("noi nhan", "tm.", "kt.", "tl.", "tuq.", "thu ky", "nguoi ky")):
        return True
    if key in {"bo truong", "thu truong", "thu tuong", "pho thu tuong", "chu tich", "pho chu tich", "giam doc", "tong giam doc"}:
        return True
    if key.startswith(("- luu:", "+ luu:", "luu:")):
        return True
    return False


def _is_noise_after_footer(line: str) -> bool:
    key = _ascii_key(line).strip(" :-")
    if not key:
        return True
    if _is_footer_start(line):
        return True
    if key in {"da ky", "signed", "ky ten", "dau"}:
        return True
    if re.fullmatch(r"[A-Z\s\.\-]{5,}", _norm_space(line)):
        return True
    return False


def _is_structural_restart(line: str) -> bool:
    key = _ascii_key(line)
    return bool(APPENDIX_RE.match(key) or UPPER_SECTION_RE.match(key) or ARTICLE_RE.match(key))


def _append_text(node: LegalNode, line: str) -> None:
    node.text = _norm_space(f"{node.text}\n{line}")


def _node(
    *,
    doc_id: str,
    node_type: str,
    label: str,
    text: str,
    parent: LegalNode,
    order: int,
    article: Optional[str] = None,
    clause: Optional[str] = None,
    point: Optional[str] = None,
    item: Optional[str] = None,
) -> LegalNode:
    return LegalNode(
        node_id=_new_node_id(doc_id, node_type, order, parent.label, label),
        doc_id=doc_id,
        node_type=node_type,
        label=_norm_space(label),
        text=_norm_space(text),
        parent_id=parent.node_id,
        order_index=order,
        level=parent.level + 1,
        article=article,
        clause=clause,
        point=point,
        item=item,
    )


def _register(node: LegalNode, parent: LegalNode, nodes: List[LegalNode], node_index: Dict[str, LegalNode]) -> LegalNode:
    nodes.append(node)
    node_index[node.node_id] = node
    parent.children_ids.append(node.node_id)
    return node


def _is_roman_section_candidate(line: str, *, current_article: Optional[LegalNode], current_appendix: Optional[LegalNode]) -> bool:
    if not ROMAN_SECTION_RE.match(line):
        return False
    if current_article is not None and current_appendix is None:
        return False
    return True


def _is_alpha_section_candidate(line: str, *, current_article: Optional[LegalNode], current_appendix: Optional[LegalNode]) -> bool:
    if not ALPHA_SECTION_RE.match(line):
        return False
    if current_article is not None and current_appendix is None:
        return False
    title = ALPHA_SECTION_RE.match(line).group("title")
    title_key = _ascii_key(title)
    return bool(current_appendix or len(title_key) >= 8)


def _table_parent(
    doc_node: LegalNode,
    current_appendix: Optional[LegalNode],
    current_section: Optional[LegalNode],
    current_article: Optional[LegalNode],
    current_clause: Optional[LegalNode],
    current_point: Optional[LegalNode],
    current_item: Optional[LegalNode],
) -> LegalNode:
    for candidate in (current_point, current_clause, current_item, current_article, current_appendix, current_section):
        if candidate is not None and candidate.node_type in TABLE_PARENT_TYPES:
            return candidate
    return doc_node


def _roman_section_parent(
    doc_node: LegalNode,
    current_appendix: Optional[LegalNode],
    current_section: Optional[LegalNode],
    node_index: Dict[str, LegalNode],
) -> LegalNode:
    if current_appendix is not None:
        return current_appendix
    if current_section is not None and current_section.node_type in {"roman_section", "alpha_section"}:
        return node_index.get(current_section.parent_id or "", doc_node)
    return current_section or doc_node


def _alpha_section_parent(
    doc_node: LegalNode,
    current_appendix: Optional[LegalNode],
    current_section: Optional[LegalNode],
    node_index: Dict[str, LegalNode],
) -> LegalNode:
    if current_appendix is not None:
        return current_appendix
    if current_section is not None and current_section.node_type == "alpha_section":
        return node_index.get(current_section.parent_id or "", doc_node)
    return current_section or doc_node


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

    current_section: Optional[LegalNode] = None
    current_appendix: Optional[LegalNode] = None
    current_article: Optional[LegalNode] = None
    current_clause: Optional[LegalNode] = None
    current_point: Optional[LegalNode] = None
    current_item: Optional[LegalNode] = None
    current_table: Optional[LegalNode] = None
    current_anchor: LegalNode = doc_node
    footer_mode = False
    order = 1

    for line in body_lines:
        key = _ascii_key(line)

        if footer_mode:
            if _is_structural_restart(line) or _looks_like_table_line(line):
                footer_mode = False
            elif _is_noise_after_footer(line):
                continue
            else:
                continue

        if _is_footer_start(line):
            footer_mode = True
            continue

        current_table = None if not _looks_like_table_line(line) else current_table

        if APPENDIX_RE.match(key):
            parent = doc_node
            label = line
            node = _node(doc_id=header.doc_id, node_type="appendix", label=label, text=line, parent=parent, order=order)
            order += 1
            _register(node, parent, nodes, node_index)
            current_section = node
            current_appendix = node
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        if ATTACHMENT_RE.match(key):
            parent = doc_node
            node = _node(doc_id=header.doc_id, node_type="attachment", label=line, text=line, parent=parent, order=order)
            order += 1
            _register(node, parent, nodes, node_index)
            current_section = node
            current_appendix = None
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        if UPPER_SECTION_RE.match(key):
            parent = current_appendix or doc_node
            node = _node(doc_id=header.doc_id, node_type="section", label=line, text=line, parent=parent, order=order)
            order += 1
            _register(node, parent, nodes, node_index)
            current_section = node
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        article_m = ARTICLE_RE.match(key)
        if article_m:
            article_num = article_m.group(1)
            suffix = _norm_space(re.sub(r"^\S+\s+\S+[\.:;\-\)]?\s*", "", line, count=1))
            article = f"{ARTICLE_LABEL} {article_num}"
            label = article if not suffix else f"{article}. {suffix}"
            parent = current_appendix or doc_node
            node = _node(doc_id=header.doc_id, node_type="article", label=label, text=line, parent=parent, order=order, article=article)
            order += 1
            _register(node, parent, nodes, node_index)
            current_article = node
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        if _is_roman_section_candidate(line, current_article=current_article, current_appendix=current_appendix):
            m = ROMAN_SECTION_RE.match(line)
            label = f"{m.group('label')}. {_norm_space(m.group('title'))}"
            parent = _roman_section_parent(doc_node, current_appendix, current_section, node_index)
            node = _node(doc_id=header.doc_id, node_type="roman_section", label=label, text=line, parent=parent, order=order)
            order += 1
            _register(node, parent, nodes, node_index)
            current_section = node
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        if _is_alpha_section_candidate(line, current_article=current_article, current_appendix=current_appendix):
            m = ALPHA_SECTION_RE.match(line)
            label = f"{m.group('label')}. {_norm_space(m.group('title'))}"
            parent = _alpha_section_parent(doc_node, current_appendix, current_section, node_index)
            node = _node(doc_id=header.doc_id, node_type="alpha_section", label=label, text=line, parent=parent, order=order)
            order += 1
            _register(node, parent, nodes, node_index)
            current_section = node
            current_article = None
            current_clause = None
            current_point = None
            current_item = None
            current_table = None
            current_anchor = node
            continue

        if _looks_like_table_line(line):
            parent = _table_parent(doc_node, current_appendix, current_section, current_article, current_clause, current_point, current_item)
            if current_table is None or current_table.parent_id != parent.node_id:
                current_table = _node(doc_id=header.doc_id, node_type="table", label="Bảng", text="", parent=parent, order=order)
                order += 1
                _register(current_table, parent, nodes, node_index)
            row_label = f"Dòng bảng {len(current_table.children_ids) + 1}"
            row = _node(
                doc_id=header.doc_id,
                node_type="table_row",
                label=row_label,
                text=line,
                parent=current_table,
                order=order,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=current_point.point if current_point else None,
                item=current_item.item if current_item else None,
            )
            order += 1
            _append_text(current_table, line)
            _register(row, current_table, nodes, node_index)
            current_anchor = row
            continue

        current_table = None

        decimal_m = DECIMAL_ITEM_RE.match(line)
        if decimal_m:
            item_num = decimal_m.group("label")
            label = item_num
            parent = current_item if current_item and current_item.node_type in ITEM_LIKE_TYPES else (current_section or current_appendix or doc_node)
            node = _node(doc_id=header.doc_id, node_type="decimal_item", label=label, text=line, parent=parent, order=order, item=item_num)
            order += 1
            _register(node, parent, nodes, node_index)
            current_item = node
            current_clause = None
            current_point = None
            current_anchor = node
            continue

        numbered_m = NUMBERED_ITEM_RE.match(line)
        if numbered_m and current_article is not None:
            clause_num = numbered_m.group("label")
            clause = f"{CLAUSE_LABEL} {clause_num}"
            node = _node(
                doc_id=header.doc_id,
                node_type="clause",
                label=clause,
                text=line,
                parent=current_article,
                order=order,
                article=current_article.article,
                clause=clause,
            )
            order += 1
            _register(node, current_article, nodes, node_index)
            current_clause = node
            current_point = None
            current_item = None
            current_anchor = node
            continue

        if numbered_m and current_article is None:
            item_num = numbered_m.group("label")
            parent = current_section or current_appendix or doc_node
            node = _node(doc_id=header.doc_id, node_type="item", label=item_num, text=line, parent=parent, order=order, item=item_num)
            order += 1
            _register(node, parent, nodes, node_index)
            current_item = node
            current_clause = None
            current_point = None
            current_anchor = node
            continue

        point_m = POINT_RE.match(line)
        if point_m and (current_clause is not None or current_item is not None):
            ch = point_m.group("label").lower()
            point = f"{POINT_LABEL} {ch}"
            parent = current_clause or current_item
            node = _node(
                doc_id=header.doc_id,
                node_type="point",
                label=point,
                text=line,
                parent=parent,
                order=order,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=point,
                item=current_item.item if current_item else None,
            )
            order += 1
            _register(node, parent, nodes, node_index)
            current_point = node
            current_anchor = node
            continue

        list_m = NUMBER_PAREN_RE.match(line) or PAREN_ITEM_RE.match(line)
        if list_m:
            token = str(list_m.group("label")).lower()
            parent = current_point or current_clause or current_item or current_article or current_section or current_appendix or doc_node
            node = _node(
                doc_id=header.doc_id,
                node_type="list_item",
                label=token,
                text=line,
                parent=parent,
                order=order,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=current_point.point if current_point else None,
                item=current_item.item if current_item else token,
            )
            order += 1
            _register(node, parent, nodes, node_index)
            current_item = node if current_article is None else current_item
            current_anchor = node
            continue

        bullet_m = BULLET_RE.match(line)
        if bullet_m:
            parent = current_point or current_clause or current_item or current_article or current_section or current_appendix or doc_node
            node = _node(
                doc_id=header.doc_id,
                node_type="bullet",
                label="Bullet",
                text=line,
                parent=parent,
                order=order,
                article=current_article.article if current_article else None,
                clause=current_clause.clause if current_clause else None,
                point=current_point.point if current_point else None,
                item=current_item.item if current_item else None,
            )
            order += 1
            _register(node, parent, nodes, node_index)
            current_anchor = node
            continue

        _append_text(current_anchor, line)

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


def _doc_sketch_text(header: DocumentHeader, major_labels: List[str]) -> str:
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
    if major_labels:
        preview = major_labels[:15]
        parts.append("Các phần chính:\n" + "\n".join(f"- {x}" for x in preview))
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


def _base_metadata(header: DocumentHeader, node: LegalNode, parent: Optional[LegalNode], sibling_ids: List[str], heading_title: str) -> Dict[str, object]:
    return {
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


def build_chunks(parsed: ParsedLegalDocument) -> List[dict]:
    header = parsed.header
    nodes = parsed.nodes
    node_index = {node.node_id: node for node in nodes}
    chunks: List[dict] = []

    article_nodes: List[LegalNode] = []
    section_bundle_roots: List[LegalNode] = []

    for node in nodes:
        if node.node_type == "document":
            continue
        parent = node_index.get(node.parent_id) if node.parent_id else None
        sibling_ids = parent.children_ids if parent else []
        sibling_ids = [x for x in sibling_ids if x != node.node_id]
        heading_title = _first_line(node.text) or node.label
        metadata = _base_metadata(header, node, parent, sibling_ids, heading_title)
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
            **_base_metadata(header, article, None, [], article.label),
            "artifact_type": "article_bundle",
            "node_id": bundle_id,
            "chunk_id": bundle_id,
            "node_type": "article_bundle",
            "title": article.label,
            "heading_title": article.label,
            "path_title": article.path_title or article.label,
            "parent_id": None,
            "children_ids": [],
            "sibling_ids": [],
            "source_node_ids": source_ids,
            "level": article.level,
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
                f"Vị trí pháp lý: {article.path_title or article.label}",
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

    section_bundle_ids: List[str] = []
    for root in section_bundle_roots:
        descendant_ids = _descendant_ids(root.node_id, node_index)
        source_ids = [root.node_id] + descendant_ids
        source_nodes = [node_index[nid] for nid in source_ids if nid in node_index]
        bundle_lines = [_norm_space(n.text) for n in source_nodes if _norm_space(n.text)]
        bundle_text = "\n".join(bundle_lines).strip()
        if not bundle_text:
            continue
        bundle_id = f"{root.node_id}::section_bundle"
        section_bundle_ids.append(bundle_id)
        metadata = {
            **_base_metadata(header, root, None, [], root.label),
            "artifact_type": "section_bundle",
            "node_id": bundle_id,
            "chunk_id": bundle_id,
            "node_type": "section_bundle",
            "title": root.label,
            "heading_title": root.label,
            "path_title": root.path_title or root.label,
            "parent_id": None,
            "children_ids": [],
            "sibling_ids": [],
            "source_node_ids": source_ids,
            "level": root.level,
        }
        retrieval_text = "\n".join(
            [
                f"Văn bản: {header.official_title or header.file_stem}",
                f"Mục/Nhóm ý: {root.path_title or root.label}",
                f"Nội dung nhóm: {bundle_text}",
            ]
        ).strip()
        rerank_text = "\n".join(
            [
                f"Văn bản: {header.official_title or header.file_stem}",
                f"Vị trí pháp lý: {root.path_title or root.label}",
                f"Nội dung nhóm: {_truncate(bundle_text, 1800)}",
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
    major_nodes = [n for n in nodes if n.node_type in SECTION_LIKE_TYPES | {"article"} and n.parent_id == f"{header.doc_id}::document"]
    if not major_nodes:
        major_nodes = [n for n in nodes if n.node_type in SECTION_LIKE_TYPES | {"article"}]
    major_labels = [n.path_title or n.label for n in sorted(major_nodes, key=lambda x: x.order_index)]
    doc_sketch_text = _doc_sketch_text(header, major_labels)
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
            "item": None,
            "parent_id": None,
            "children_ids": [],
            "sibling_ids": [],
            "source_node_ids": article_bundle_ids + section_bundle_ids,
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
