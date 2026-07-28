from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

ARTICLE_LABEL = "\u0110i\u1ec1u"
CLAUSE_LABEL = "Kho\u1ea3n"
POINT_LABEL = "\u0110i\u1ec3m"

ARTICLE_NUM_RE = re.compile(r"\bdieu\s*(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_NUM_RE = re.compile(r"\bkhoan\s*(\d+)\b", re.IGNORECASE)
POINT_RE = re.compile(r"\bdiem\s*([a-z]|d)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/(?:\d{4}/)?[A-ZĐD\-]+\b", re.IGNORECASE)

LAW_TYPE_PATTERNS = {
    "bo luat": "B\u1ed9 lu\u1eadt",
    "luat": "Lu\u1eadt",
    "nghi dinh": "Ngh\u1ecb \u0111\u1ecbnh",
    "thong tu": "Th\u00f4ng t\u01b0",
    "quyet dinh": "Quy\u1ebft \u0111\u1ecbnh",
    "chi thi": "Ch\u1ec9 th\u1ecb",
    "cong dien": "C\u00f4ng \u0111i\u1ec7n",
    "nghi quyet": "Ngh\u1ecb quy\u1ebft",
}

HEADING_TERMS = [
    "nguyen tac",
    "quan diem",
    "muc tieu",
    "yeu cau",
    "pham vi",
    "doi tuong ap dung",
    "trach nhiem",
    "nhiem vu",
    "giai phap",
    "to chuc thuc hien",
    "phuong thuc",
    "dieu kien",
    "ho so",
    "thoi han",
    "trinh tu",
    "thu tuc",
    "hieu luc thi hanh",
]

CHANGE_TERMS = [
    "bai bo",
    "sua doi",
    "bo sung",
    "thay the",
    "co hieu luc",
    "het hieu luc",
]

CONDITION_TERMS = [
    "truong hop nao",
    "khi nao",
    "dieu kien nao",
    "duoc phan cap",
    "duoc",
    "phai",
    "neu",
]

LIST_TRIGGERS = [
    "bao gom nhung gi",
    "gom nhung gi",
    "gom nhung noi dung nao",
    "nhung nguyen tac nao",
    "nhung truong hop nao",
    "nhung dieu kien nao",
    "liet ke",
]

CONFLICTING_HEADING_TERMS = {
    "nguyen tac": ["phuong thuc", "trach nhiem", "dieu kien", "thoi han", "doi tuong ap dung"],
    "phuong thuc": ["nguyen tac", "trach nhiem", "dieu kien", "thoi han"],
    "trach nhiem": ["nguyen tac", "phuong thuc", "dieu kien", "thoi han"],
    "dieu kien": ["nguyen tac", "phuong thuc", "trach nhiem", "thoi han"],
    "thoi han": ["nguyen tac", "phuong thuc", "trach nhiem", "dieu kien"],
}


def _repair_mojibake(text: Any) -> str:
    clean = str(text or "")
    markers = ("Ã", "Ä", "Â", "áº", "á»", "�")
    for _ in range(2):
        if not any(marker in clean for marker in markers):
            break
        try:
            repaired = clean.encode("latin1").decode("utf-8")
        except Exception:
            break
        if repaired == clean:
            break
        clean = repaired
    return clean


def _norm_space(text: Any) -> str:
    return " ".join(_repair_mojibake(text).split()).strip()


def _ascii_key(text: Any) -> str:
    raw = unicodedata.normalize("NFD", _norm_space(text))
    raw = "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")
    raw = raw.replace("\u0111", "d").replace("\u0110", "D")
    return raw.lower()


def _contains_any(q_key: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if p in q_key]


def is_likely_document_year_filter(question: str, year: Any) -> bool:
    year_text = str(year or "").strip()
    if not year_text:
        return False
    q_key = _ascii_key(question)
    idx = q_key.find(year_text)
    if idx < 0:
        return False

    horizon_markers = (
        "den nam",
        "tam nhin",
        "giai doan",
        "ke hoach nam",
        "du toan ngan sach",
        "quy hoach",
        "muc tieu",
    )
    local = q_key[max(0, idx - 55) : idx + len(year_text) + 30]
    if any(marker in local for marker in horizon_markers):
        return False

    document_year_markers = (
        "ban hanh",
        "ngay",
        "thang",
        "nam ban hanh",
        "van ban nam",
        "hieu luc",
    )
    return any(marker in local for marker in document_year_markers)


def infer_filters(question: str) -> Dict[str, Any]:
    q = _norm_space(question)
    q_key = _ascii_key(q)
    out: Dict[str, Any] = {}

    m = ARTICLE_NUM_RE.search(q_key)
    if m:
        out["article"] = f"{ARTICLE_LABEL} {m.group(1)}"

    m = CLAUSE_NUM_RE.search(q_key)
    if m:
        out["clause"] = f"{CLAUSE_LABEL} {m.group(1)}"

    m = POINT_RE.search(q_key)
    if m:
        raw_point = m.group(1).lower()
        out["point"] = f"{POINT_LABEL} {'\u0111' if raw_point == 'd' else raw_point}"

    y = YEAR_RE.search(q_key)
    if y and is_likely_document_year_filter(q, y.group(1)):
        try:
            out["year"] = int(y.group(1))
        except Exception:
            pass

    doc_no = DOC_NUMBER_RE.search(q.upper())
    if doc_no:
        out["doc_number"] = doc_no.group(0).upper()

    for key, value in LAW_TYPE_PATTERNS.items():
        if key in q_key:
            out["law_type"] = value
            break

    return out


def infer_query_profile(question: str) -> Dict[str, Any]:
    q = _norm_space(question)
    q_key = _ascii_key(q)
    filters = infer_filters(q)

    heading_terms = _contains_any(q_key, HEADING_TERMS)
    change_terms = _contains_any(q_key, CHANGE_TERMS)
    condition_terms = _contains_any(q_key, CONDITION_TERMS)
    list_terms = _contains_any(q_key, LIST_TRIGGERS)

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
    }
