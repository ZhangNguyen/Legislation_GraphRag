from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

ARTICLE_NUM_RE = re.compile(r"(?:điều)\s*(\d+)", re.IGNORECASE)
CLAUSE_NUM_RE = re.compile(r"(?:khoản)\s*(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"(?:điểm)\s*([a-zđ])", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
DOC_NUMBER_RE = re.compile(r"\b\d{1,3}/\d{4}/[A-ZĐ\-]+\b", re.IGNORECASE)

LAW_TYPE_PATTERNS = {
    "bộ luật": "Bộ luật",
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
    "chỉ thị": "Chỉ thị",
    "công điện": "Công điện",
    "nghị quyết": "Nghị quyết",
}

HEADING_TERMS = [
    "nguyên tắc",
    "phạm vi",
    "đối tượng áp dụng",
    "trách nhiệm",
    "phương thức",
    "điều kiện",
    "hồ sơ",
    "thời hạn",
    "trình tự",
    "thủ tục",
    "hiệu lực thi hành",
]

CHANGE_TERMS = [
    "bãi bỏ",
    "sửa đổi",
    "bổ sung",
    "thay thế",
    "có hiệu lực",
    "hết hiệu lực",
]

CONDITION_TERMS = [
    "trường hợp nào",
    "khi nào",
    "điều kiện nào",
    "được phân cấp",
    "nếu",
]

LIST_TRIGGERS = [
    "bao gồm những gì",
    "gồm những gì",
    "gồm những nội dung nào",
    "những nguyên tắc nào",
    "những trường hợp nào",
    "những điều kiện nào",
    "liệt kê",
    "làm gì",
    "phải làm gì",
    "cần làm gì",
    "thực hiện gì",
    "thực hiện những gì",
]


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _contains_any(q: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if p in q]


CONFLICTING_HEADING_TERMS = {
    "nguyên tắc": ["phương thức", "trách nhiệm", "điều kiện", "thời hạn", "đối tượng áp dụng"],
    "phương thức": ["nguyên tắc", "trách nhiệm", "điều kiện", "thời hạn"],
    "trách nhiệm": ["nguyên tắc", "phương thức", "điều kiện", "thời hạn"],
    "điều kiện": ["nguyên tắc", "phương thức", "trách nhiệm", "thời hạn"],
    "thời hạn": ["nguyên tắc", "phương thức", "trách nhiệm", "điều kiện"],
}


def infer_filters(question: str) -> Dict[str, Any]:
    q = _norm_space(str(question or ""))
    q_lower = q.lower()
    out: Dict[str, Any] = {}

    m = ARTICLE_NUM_RE.search(q)
    if m:
        out["article"] = f"Điều {m.group(1)}"

    m = CLAUSE_NUM_RE.search(q)
    if m:
        out["clause"] = f"Khoản {m.group(1)}"

    m = POINT_RE.search(q)
    if m:
        out["point"] = f"Điểm {m.group(1).lower()}"

    y = YEAR_RE.search(q)
    if y:
        try:
            out["year"] = int(y.group(1))
        except Exception:
            pass

    doc_no = DOC_NUMBER_RE.search(q)
    if doc_no:
        out["doc_number"] = doc_no.group(0).upper()

    for key, value in LAW_TYPE_PATTERNS.items():
        if key in q_lower:
            out["law_type"] = value
            break

    return out


def infer_query_profile(question: str) -> Dict[str, Any]:
    q = _norm_space(str(question or ""))
    q_lower = q.lower()
    filters = infer_filters(q)

    heading_terms = _contains_any(q_lower, HEADING_TERMS)
    change_terms = _contains_any(q_lower, CHANGE_TERMS)
    condition_terms = _contains_any(q_lower, CONDITION_TERMS)
    list_terms = _contains_any(q_lower, LIST_TRIGGERS)
    asks_responsibility = any(
        phrase in q_lower
        for phrase in [
            "ai chịu trách nhiệm",
            "trách nhiệm của",
            "ủy ban nhân dân",
            "ubnd",
            "bộ công thương",
            "cơ quan nào",
            "làm gì",
            "thực hiện gì",
        ]
    )

    explicit_ref = any(k in filters for k in ["article", "clause", "point"])
    has_doc_number = "doc_number" in filters

    if change_terms:
        route = "version_change"
    elif explicit_ref:
        route = "direct_reference"
    elif heading_terms and list_terms:
        route = "heading_list"
    elif heading_terms and not condition_terms:
        route = "heading_list"
    elif asks_responsibility and (list_terms or " để " in f" {q_lower} "):
        route = "heading_list"
    elif condition_terms:
        route = "condition_circumstance"
    elif has_doc_number:
        route = "document_focus"
    else:
        route = "factoid"

    primary_heading_term: Optional[str] = heading_terms[0] if heading_terms else None
    conflicting = CONFLICTING_HEADING_TERMS.get(primary_heading_term or "", [])

    return {
        "question": q,
        "route": route,
        "filters": filters,
        "intent_terms": sorted(set(heading_terms + change_terms + list_terms)),
        "heading_terms": heading_terms,
        "change_terms": change_terms,
        "condition_terms": condition_terms,
        "list_terms": list_terms,
        "primary_heading_term": primary_heading_term,
        "conflicting_heading_terms": conflicting,
        "explicit_reference": explicit_ref,
        "has_doc_number": has_doc_number,
        "prefers_article_bundle": route in {"heading_list", "version_change"} or (explicit_ref and "clause" not in filters and "point" not in filters),
        "wants_list_answer": bool(list_terms or route == "heading_list"),
        "asks_responsibility": asks_responsibility,
    }
