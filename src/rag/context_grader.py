from __future__ import annotations

import re
from typing import Any, Dict, List

from src.rag.retrieval_pipeline_simple import analyze_query

_LEGAL_REASONING_CUES = [
    "ai có lỗi",
    "bên nào sai",
    "ai chịu trách nhiệm",
    "trách nhiệm thuộc về ai",
    "lỗi trong va chạm",
]
_RELEVANCE_TERMS = [
    "mức phạt",
    "xử phạt",
    "thủ tục",
    "điều kiện",
    "trách nhiệm",
    "hiệu lực",
    "không chấp hành",
    "tín hiệu giao thông",
    "đèn đỏ",
    "va chạm",
]


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _norm_key(text: Any) -> str:
    return _norm_space(text).lower()


def _source_text(passages: List[Dict[str, Any]]) -> str:
    return "\n".join(_norm_space(p.get("text") or p.get("local_text") or p.get("snippet") or "") for p in passages).lower()


def _metadata_matches(passages: List[Dict[str, Any]], key: str, expected: Any) -> bool:
    if expected in (None, ""):
        return True
    aliases = {
        "law_type": ("law_type", "doc_type"),
        "doc_number": ("doc_number",),
        "article": ("article",),
        "clause": ("clause",),
        "point": ("point",),
    }
    for passage in passages:
        md = dict(passage.get("metadata") or {})
        for candidate_key in aliases.get(key, (key,)):
            value = md.get(candidate_key)
            if value in (None, ""):
                continue
            if _norm_key(value) == _norm_key(expected):
                return True
    return False


def _has_relevant_text(question: str, passages: List[Dict[str, Any]]) -> bool:
    hay = _source_text(passages)
    q = _norm_key(question)
    if not hay:
        return False
    if any(term in hay for term in _RELEVANCE_TERMS if term in q or term in hay):
        return True
    tokens = [tok for tok in re.findall(r"\w+", q, flags=re.UNICODE) if len(tok) >= 4]
    if not tokens:
        return bool(hay)
    hits = sum(1 for tok in set(tokens) if tok in hay)
    return hits >= max(1, min(3, len(set(tokens)) // 4))


def _needs_legal_reasoning(question: str) -> bool:
    q = _norm_key(question)
    return any(cue in q for cue in _LEGAL_REASONING_CUES)


def grade_context(question: str, query_profile: Dict[str, Any], passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not query_profile:
        query_profile = analyze_query(question)

    base = {
        "status": "insufficient",
        "reason": "",
        "matched_doc_number": False,
        "matched_article": False,
        "matched_clause": False,
        "matched_point": False,
        "has_relevant_text": False,
        "allow_reasoning": False,
    }

    if not passages:
        base["reason"] = "no_passages"
        return base

    base["matched_doc_number"] = _metadata_matches(passages, "doc_number", query_profile.get("doc_number"))
    base["matched_article"] = _metadata_matches(passages, "article", query_profile.get("article"))
    base["matched_clause"] = _metadata_matches(passages, "clause", query_profile.get("clause"))
    base["matched_point"] = _metadata_matches(passages, "point", query_profile.get("point"))
    base["has_relevant_text"] = _has_relevant_text(question, passages)

    if query_profile.get("doc_number") and not base["matched_doc_number"]:
        base["reason"] = "requested_doc_number_not_found"
        return base
    if query_profile.get("article") and not base["matched_article"]:
        base["reason"] = "requested_article_not_found"
        return base
    if query_profile.get("clause") and not base["matched_clause"]:
        base["reason"] = "requested_clause_not_found"
        return base
    if query_profile.get("point") and not base["matched_point"]:
        base["reason"] = "requested_point_not_found"
        return base

    if _needs_legal_reasoning(question) and base["has_relevant_text"]:
        base["status"] = "needs_reasoning"
        base["allow_reasoning"] = True
        base["reason"] = "situation_question_with_relevant_rules"
        return base

    if base["has_relevant_text"]:
        base["status"] = "sufficient"
        base["allow_reasoning"] = False
        base["reason"] = "matched_retrieved_context"
        return base

    base["status"] = "insufficient"
    base["reason"] = "no_relevant_text"
    return base
