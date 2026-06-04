from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from src.evaluation.schemas import REFERENCE_FIELDS, norm_key, norm_text

DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+|\b\d{1,4}/VBHN-[A-ZĐ\-]+", re.IGNORECASE)
ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)

WEIGHTS = {
    "doc_number": 0.40,
    "law_type": 0.10,
    "year": 0.10,
    "article": 0.20,
    "clause": 0.15,
    "point": 0.05,
}

STOPWORDS = {
    "và",
    "của",
    "các",
    "những",
    "trong",
    "cho",
    "về",
    "theo",
    "là",
    "có",
    "được",
    "phải",
    "này",
    "đó",
    "với",
    "một",
    "hoặc",
    "để",
    "tại",
}


def _expected_refs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(x) for x in list(sample.get("expected_context_refs") or []) if isinstance(x, dict)]


def _context_metadata(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(x) for x in list(sample.get("context_metadata") or []) if isinstance(x, dict)]


def _norm_doc_number(value: Any) -> str:
    raw = norm_text(value).upper().replace("Đ", "D")
    parts = raw.split("/")
    if parts and parts[0].isdigit():
        parts[0] = str(int(parts[0]))
    return "/".join(parts)


def _tokens(value: Any) -> Set[str]:
    words = re.findall(r"\w+", norm_key(value), flags=re.UNICODE)
    return {word for word in words if len(word) >= 3 and word not in STOPWORDS}


def _gold_text(sample: Dict[str, Any]) -> str:
    return "\n".join(
        norm_text(sample.get(key))
        for key in ("ground_truth", "reference", "source_excerpt")
        if norm_text(sample.get(key))
    )


def _field_match(expected: Dict[str, Any], context_metadata: List[Dict[str, Any]], field: str) -> bool:
    value = norm_key(expected.get(field))
    if not value:
        return True
    for md in context_metadata:
        if field == "doc_number" and _norm_doc_number(md.get(field)) == _norm_doc_number(expected.get(field)):
            return True
        actual = norm_key(md.get(field))
        if actual == value:
            return True
        if field == "law_type" and norm_key(md.get("doc_type")) == value:
            return True
    return False


def context_sufficiency_lite(sample: dict) -> dict:
    contexts = list(sample.get("contexts") or [])
    if not contexts:
        return {"score": 0.0, "matched_fields": [], "missing_fields": [], "warnings": ["No contexts."]}

    refs = _expected_refs(sample)
    expected_fields = []
    for ref in refs:
        expected_fields.extend([field for field in REFERENCE_FIELDS if norm_text(ref.get(field))])
    expected_fields = list(dict.fromkeys(expected_fields))
    if not expected_fields:
        return {
            "score": None,
            "matched_fields": [],
            "missing_fields": [],
            "warnings": ["No non-empty expected_context_refs fields."],
        }

    metadata = _context_metadata(sample)
    matched: List[str] = []
    missing: List[str] = []
    for field in expected_fields:
        if any(_field_match(ref, metadata, field) for ref in refs):
            matched.append(field)
        else:
            missing.append(field)
    return {
        "score": len(matched) / max(1, len(expected_fields)),
        "matched_fields": matched,
        "missing_fields": missing,
        "warnings": [],
    }


def retrieval_ref_match_lite(sample: dict) -> dict:
    refs = _expected_refs(sample)
    metadata = _context_metadata(sample)
    expected_fields = []
    for ref in refs:
        expected_fields.extend([field for field in REFERENCE_FIELDS if norm_text(ref.get(field))])
    expected_fields = list(dict.fromkeys(expected_fields))
    if not expected_fields:
        return {"score": None, "matched": {field: None for field in REFERENCE_FIELDS}, "reason": "No expected fields."}

    total_weight = sum(WEIGHTS[field] for field in expected_fields)
    score = 0.0
    matched: Dict[str, Any] = {field: None for field in REFERENCE_FIELDS}
    for field in expected_fields:
        ok = any(_field_match(ref, metadata, field) for ref in refs)
        matched[field] = ok
        if ok:
            score += WEIGHTS[field]
    normalized = score / max(total_weight, 1e-9)
    hit = [field for field, ok in matched.items() if ok is True]
    miss = [field for field, ok in matched.items() if ok is False]
    return {
        "score": normalized,
        "matched": matched,
        "reason": f"Matched {', '.join(hit) or 'none'}; missing {', '.join(miss) or 'none'}.",
    }


def _context_matches_expected_ref(context_md: Dict[str, Any], refs: List[Dict[str, Any]]) -> bool:
    if not refs:
        return False
    for ref in refs:
        expected_fields = [field for field in REFERENCE_FIELDS if norm_text(ref.get(field))]
        if not expected_fields:
            continue
        if all(_field_match({field: ref.get(field)}, [context_md], field) for field in expected_fields):
            return True
    return False


def context_precision_lite(sample: dict) -> dict:
    contexts = [norm_text(ctx) for ctx in list(sample.get("contexts") or []) if norm_text(ctx)]
    if not contexts:
        return {"score": 0.0, "relevant_contexts": 0, "total_contexts": 0, "warnings": ["No contexts."]}

    refs = _expected_refs(sample)
    metadata = _context_metadata(sample)
    gold_tokens = _tokens(_gold_text(sample))
    relevant = 0
    details: List[Dict[str, Any]] = []

    for idx, context in enumerate(contexts):
        md = metadata[idx] if idx < len(metadata) else {}
        metadata_hit = _context_matches_expected_ref(md, refs)
        context_tokens = _tokens(context)
        token_hits = len(context_tokens & gold_tokens)
        overlap = token_hits / max(1, len(gold_tokens)) if gold_tokens else 0.0
        is_relevant = metadata_hit or (token_hits >= 2 and overlap >= 0.18)
        if is_relevant:
            relevant += 1
        details.append(
            {
                "index": idx + 1,
                "relevant": is_relevant,
                "metadata_hit": metadata_hit,
                "gold_token_hits": token_hits,
                "gold_token_overlap": round(overlap, 4),
            }
        )

    return {
        "score": relevant / max(1, len(contexts)),
        "relevant_contexts": relevant,
        "total_contexts": len(contexts),
        "details": details,
        "warnings": [] if gold_tokens or refs else ["No ground_truth/reference/source_excerpt or expected refs."],
    }


def context_recall_lite(sample: dict) -> dict:
    contexts = [norm_text(ctx) for ctx in list(sample.get("contexts") or []) if norm_text(ctx)]
    if not contexts:
        return {"score": 0.0, "token_recall": 0.0, "metadata_recall": 0.0, "warnings": ["No contexts."]}

    combined_context = "\n".join(contexts)
    gold_tokens = _tokens(_gold_text(sample))
    if gold_tokens:
        token_recall = len(gold_tokens & _tokens(combined_context)) / max(1, len(gold_tokens))
    else:
        token_recall = None

    sufficiency = context_sufficiency_lite(sample)
    metadata_recall = sufficiency.get("score")

    parts = [value for value in (token_recall, metadata_recall) if isinstance(value, (int, float))]
    score = sum(float(value) for value in parts) / len(parts) if parts else None
    return {
        "score": score,
        "token_recall": token_recall,
        "metadata_recall": metadata_recall,
        "warnings": [] if parts else ["No ground truth tokens or expected refs."],
    }


def _extract_answer_refs(answer: str) -> List[str]:
    refs: List[str] = []
    for m in DOC_NUMBER_RE.finditer(answer or ""):
        refs.append(m.group(0).upper())
    for m in ARTICLE_RE.finditer(answer or ""):
        refs.append(f"Điều {m.group(1)}")
    for m in CLAUSE_RE.finditer(answer or ""):
        refs.append(f"Khoản {m.group(1)}")
    for m in POINT_RE.finditer(answer or ""):
        refs.append(f"Điểm {m.group(1).lower()}")
    return list(dict.fromkeys(refs))


def _metadata_supports(ref: str, metadata: List[Dict[str, Any]]) -> bool:
    ref_norm = norm_key(ref)
    for md in metadata:
        if "/" in ref and _norm_doc_number(md.get("doc_number")) == _norm_doc_number(ref):
            return True
        values = [
            md.get("doc_number"),
            md.get("article"),
            md.get("clause"),
            md.get("point"),
        ]
        if any(norm_key(value) == ref_norm for value in values):
            return True
    return False


def citation_support_lite(answer: str, context_metadata: list[dict]) -> dict:
    refs = _extract_answer_refs(answer)
    if not refs:
        return {
            "score": 0.5,
            "supported_references": [],
            "unsupported_references": [],
            "warnings": ["Answer has no explicit citation."],
        }
    metadata = [dict(x) for x in context_metadata if isinstance(x, dict)]
    supported = [ref for ref in refs if _metadata_supports(ref, metadata)]
    unsupported = [ref for ref in refs if ref not in supported]
    return {
        "score": len(supported) / max(1, len(refs)),
        "supported_references": supported,
        "unsupported_references": unsupported,
        "warnings": [],
    }
