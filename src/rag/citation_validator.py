from __future__ import annotations

import re
from typing import Any, Dict, List

_ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
_POINT_RE = re.compile(r"(?<!quan\s)điểm\s+([a-zđ])(?=\s|[,.;:)\]]|$)", re.IGNORECASE)
_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+\b", re.IGNORECASE)


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _norm_key(text: Any) -> str:
    return _norm_space(text).lower()


def _source_values(sources: List[Dict[str, Any]], key: str) -> set[str]:
    aliases = {
        "doc_number": ("doc_number",),
        "article": ("article",),
        "clause": ("clause",),
        "point": ("point",),
    }
    values: set[str] = set()
    for source in sources:
        md = dict(source.get("metadata") or {})
        merged = {**md, **source}
        for candidate_key in aliases.get(key, (key,)):
            value = merged.get(candidate_key)
            if value not in (None, ""):
                values.add(_norm_key(value))
        if key == "doc_number":
            source_text = " ".join(
                _norm_space(merged.get(field) or "")
                for field in ("text", "local_text", "snippet", "shared_text")
            )
            for match in _DOC_NUMBER_RE.finditer(source_text.upper()):
                values.add(_norm_key(match.group(0).upper()))
    return values


def _refs_from_answer(answer: str) -> Dict[str, List[str]]:
    text = _norm_space(answer)
    return {
        "article": [f"Điều {m.group(1)}" for m in _ARTICLE_RE.finditer(text)],
        "clause": [f"Khoản {m.group(1)}" for m in _CLAUSE_RE.finditer(text)],
        "point": [f"Điểm {m.group(1).lower()}" for m in _POINT_RE.finditer(text)],
        "doc_number": [m.group(0).upper() for m in _DOC_NUMBER_RE.finditer(text)],
    }


def validate_citations(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    refs = _refs_from_answer(answer)
    unsupported: List[Dict[str, str]] = []
    warnings: List[str] = []

    for key, labels in refs.items():
        source_values = _source_values(sources, key)
        for label in labels:
            if _norm_key(label) not in source_values:
                unsupported.append({"type": key, "reference": label})

    if unsupported:
        warnings.append("Answer references legal citations not present in source metadata.")

    return {
        "valid": not unsupported,
        "warnings": warnings,
        "unsupported_references": unsupported,
    }
