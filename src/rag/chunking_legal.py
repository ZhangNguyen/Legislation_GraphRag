from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LegalNode:
    node_id: str
    node_type: str
    section: Optional[str]
    subsection: Optional[str]
    article: Optional[str]
    clause: Optional[str]
    point: Optional[str]
    text: str
    parent_id: Optional[str]
    legal_role: Optional[str] = None
    action: Optional[str] = None
    target_article: Optional[str] = None
    target_clause: Optional[str] = None
    target_point: Optional[str] = None


ARTICLE_RE = re.compile(r"^Điều\s+(\d+)[\.:]?", re.IGNORECASE)
SECTION_RE = re.compile(r"^([IVXLCDM]+)\.\s+(.+)$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
CLAUSE_RE = re.compile(r"^(\d+)\.\s+")
POINT_RE = re.compile(r"^([a-zđ])\)\s+", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\u2022]\s+")

AMENDMENT_RE = re.compile(r"(Sửa đổi|Bổ sung|Bãi bỏ|Thay thế)", re.IGNORECASE)
EFFECTIVE_RE = re.compile(r"(có hiệu lực|hiệu lực thi hành|hết hiệu lực)", re.IGNORECASE)
RESPONSIBILITY_RE = re.compile(r"(trách nhiệm thi hành|chịu trách nhiệm thi hành)", re.IGNORECASE)
TRANSITION_RE = re.compile(r"(điều khoản chuyển tiếp|chuyển tiếp)", re.IGNORECASE)
APPLICABILITY_RE = re.compile(r"(phạm vi điều chỉnh|đối tượng áp dụng)", re.IGNORECASE)

TARGET_ARTICLE_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
TARGET_CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
TARGET_POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)


SPECIAL_NODE_TYPES = {
    "amendment",
    "effective",
    "transition",
    "responsibility",
    "applicability",
}


def new_id() -> str:
    return str(uuid.uuid4())


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def detect_target(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    article = None
    clause = None
    point = None

    m = TARGET_ARTICLE_RE.search(text)
    if m:
        article = f"Điều {m.group(1)}"

    m = TARGET_CLAUSE_RE.search(text)
    if m:
        clause = f"Khoản {m.group(1)}"

    m = TARGET_POINT_RE.search(text)
    if m:
        point = f"Điểm {m.group(1).lower()}"

    return article, clause, point


def detect_special_node_type(text: str) -> tuple[Optional[str], Optional[str]]:
    raw = text or ""

    m = AMENDMENT_RE.search(raw)
    if m:
        return "amendment", m.group(1)

    if EFFECTIVE_RE.search(raw):
        return "effective", None
    if RESPONSIBILITY_RE.search(raw):
        return "responsibility", None
    if TRANSITION_RE.search(raw):
        return "transition", None
    if APPLICABILITY_RE.search(raw):
        return "applicability", None

    return None, None


def parse_legal_text(text: str) -> List[LegalNode]:
    nodes: List[LegalNode] = []

    current_section = None
    current_subsection = None
    current_article = None
    current_clause = None
    current_point = None

    section_id = None
    subsection_id = None
    article_id = None
    clause_id = None
    point_id = None
    current_anchor_id = None

    lines = [_norm_space(l) for l in str(text or "").split("\n") if _norm_space(l)]

    for line in lines:
        m = ARTICLE_RE.match(line)
        if m:
            current_article = f"Điều {m.group(1)}"
            current_clause = None
            current_point = None
            current_subsection = None

            article_id = new_id()
            clause_id = None
            point_id = None
            subsection_id = None
            current_anchor_id = article_id

            nodes.append(
                LegalNode(
                    node_id=article_id,
                    node_type="article",
                    section=current_section,
                    subsection=current_subsection,
                    article=current_article,
                    clause=None,
                    point=None,
                    text=line,
                    parent_id=section_id,
                    legal_role="article",
                )
            )
            continue

        m = SECTION_RE.match(line)
        if m:
            current_section = m.group(1).upper()
            current_subsection = None
            current_clause = None
            current_point = None

            section_id = new_id()
            subsection_id = None
            clause_id = None
            point_id = None
            current_anchor_id = section_id

            nodes.append(
                LegalNode(
                    node_id=section_id,
                    node_type="section",
                    section=current_section,
                    subsection=None,
                    article=current_article,
                    clause=None,
                    point=None,
                    text=line,
                    parent_id=article_id,
                    legal_role="section",
                )
            )
            continue

        m = SUBSECTION_RE.match(line)
        if m:
            current_subsection = m.group(1)
            current_point = None

            subsection_id = new_id()
            point_id = None
            current_anchor_id = subsection_id
            parent = section_id or clause_id or article_id

            nodes.append(
                LegalNode(
                    node_id=subsection_id,
                    node_type="subsection",
                    section=current_section,
                    subsection=current_subsection,
                    article=current_article,
                    clause=current_clause,
                    point=None,
                    text=line,
                    parent_id=parent,
                    legal_role="subsection",
                )
            )
            continue

        m = CLAUSE_RE.match(line)
        if m:
            current_clause = f"Khoản {m.group(1)}"
            current_point = None

            clause_id = new_id()
            point_id = None
            current_anchor_id = clause_id
            parent = section_id or article_id

            nodes.append(
                LegalNode(
                    node_id=clause_id,
                    node_type="clause",
                    section=current_section,
                    subsection=current_subsection,
                    article=current_article,
                    clause=current_clause,
                    point=None,
                    text=line,
                    parent_id=parent,
                    legal_role="clause",
                )
            )
            continue

        m = POINT_RE.match(line)
        if m:
            current_point = f"Điểm {m.group(1).lower()}"
            point_id = new_id()
            current_anchor_id = point_id
            parent = subsection_id or clause_id or section_id or article_id

            nodes.append(
                LegalNode(
                    node_id=point_id,
                    node_type="point",
                    section=current_section,
                    subsection=current_subsection,
                    article=current_article,
                    clause=current_clause,
                    point=current_point,
                    text=line,
                    parent_id=parent,
                    legal_role="point",
                )
            )
            continue

        special_node_type, action = detect_special_node_type(line)
        if special_node_type:
            target_article, target_clause, target_point = detect_target(line)
            parent = point_id or subsection_id or clause_id or section_id or article_id
            special_id = new_id()
            current_anchor_id = special_id

            nodes.append(
                LegalNode(
                    node_id=special_id,
                    node_type=special_node_type,
                    section=current_section,
                    subsection=current_subsection,
                    article=current_article,
                    clause=current_clause,
                    point=current_point,
                    text=line,
                    parent_id=parent,
                    legal_role=special_node_type,
                    action=action,
                    target_article=target_article,
                    target_clause=target_clause,
                    target_point=target_point,
                )
            )
            continue

        parent = current_anchor_id or point_id or subsection_id or clause_id or section_id or article_id
        node_type = "bullet" if BULLET_RE.match(line) else "text"
        legal_role = "bullet" if node_type == "bullet" else "text"
        nodes.append(
            LegalNode(
                node_id=new_id(),
                node_type=node_type,
                section=current_section,
                subsection=current_subsection,
                article=current_article,
                clause=current_clause,
                point=current_point,
                text=line,
                parent_id=parent,
                legal_role=legal_role,
            )
        )

    return nodes


def _children_by_parent(nodes: List[LegalNode]) -> Dict[str, List[LegalNode]]:
    out: Dict[str, List[LegalNode]] = {}
    for node in nodes:
        if node.parent_id:
            out.setdefault(node.parent_id, []).append(node)
    return out


def _direct_text_children(children: Dict[str, List[LegalNode]], node_id: str) -> List[str]:
    lines: List[str] = []
    for child in children.get(node_id, []):
        if child.node_type in {"text", "bullet"} and child.text.strip():
            lines.append(child.text.strip())
    return lines


def _descendant_outline(children: Dict[str, List[LegalNode]], node_id: str, *, max_items: int = 8) -> List[str]:
    queue = list(children.get(node_id, []))
    outlines: List[str] = []
    seen: set[str] = set()

    while queue and len(outlines) < max_items:
        child = queue.pop(0)
        if child.node_id in seen:
            continue
        seen.add(child.node_id)

        if child.node_type in {"article", "section", "subsection", "clause", "point", *SPECIAL_NODE_TYPES}:
            if child.text.strip():
                outlines.append(child.text.strip())

        for grand in children.get(child.node_id, []):
            if grand.node_type in {"article", "section", "subsection", "clause", "point", *SPECIAL_NODE_TYPES}:
                queue.append(grand)

    return outlines


def _path_parts(node: LegalNode) -> List[str]:
    parts: List[str] = []
    if node.section:
        parts.append(str(node.section))
    if node.subsection:
        parts.append(str(node.subsection))
    if node.article:
        parts.append(str(node.article))
    if node.clause:
        parts.append(str(node.clause))
    if node.point:
        parts.append(str(node.point))
    return parts


def _build_retrieval_text(node: LegalNode, chunk_text: str, outline: List[str]) -> str:
    header_parts = _path_parts(node)
    header_parts.append(f"node_type: {node.node_type}")
    if node.legal_role:
        header_parts.append(f"legal_role: {node.legal_role}")
    if node.action:
        header_parts.append(f"action: {node.action}")
    if node.target_article:
        header_parts.append(f"target_article: {node.target_article}")
    if node.target_clause:
        header_parts.append(f"target_clause: {node.target_clause}")
    if node.target_point:
        header_parts.append(f"target_point: {node.target_point}")

    blocks = [" | ".join(header_parts), chunk_text.strip()]
    if outline:
        blocks.append("Ngữ cảnh liên quan: " + " ; ".join(outline))
    return "\n".join(x for x in blocks if x).strip()


def build_chunks(nodes: List[LegalNode]) -> List[dict]:
    chunks: List[dict] = []
    children = _children_by_parent(nodes)

    chunkable_types = {
        "article",
        "section",
        "subsection",
        "clause",
        "point",
        "amendment",
        "effective",
        "transition",
        "responsibility",
        "applicability",
    }

    for node in nodes:
        if node.node_type not in chunkable_types:
            continue

        parts: List[str] = [node.text.strip()]
        parts.extend(_direct_text_children(children, node.node_id))

        outline_max = 10 if node.node_type == "article" else 6
        outline = _descendant_outline(children, node.node_id, max_items=outline_max)

        if node.node_type == "article" and outline:
            parts.append("Cấu trúc liên quan: " + " ; ".join(outline[:6]))

        chunk_text = "\n".join(p for p in parts if p).strip()
        retrieval_text = _build_retrieval_text(node, chunk_text, outline)

        metadata = {
            "node_id": node.node_id,
            "chunk_id": node.node_id,
            "node_type": node.node_type,
            "legal_role": node.legal_role,
            "section": node.section,
            "subsection": node.subsection,
            "article": node.article,
            "clause": node.clause,
            "point": node.point,
            "action": node.action,
            "target_article": node.target_article,
            "target_clause": node.target_clause,
            "target_point": node.target_point,
            "parent_id": node.parent_id,
            "path_title": " > ".join(_path_parts(node)),
        }

        chunks.append(
            {
                "text": chunk_text,
                "retrieval_text": retrieval_text,
                "rerank_text": retrieval_text,
                "metadata": metadata,
            }
        )

    return chunks


def legal_chunk(text: str, max_chars: int = 1600, overlap: int = 120) -> List[dict]:
    _ = (max_chars, overlap)
    nodes = parse_legal_text(text)
    return build_chunks(nodes)
