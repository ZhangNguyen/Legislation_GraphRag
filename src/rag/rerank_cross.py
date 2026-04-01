from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.app.settings import settings

logger = logging.getLogger(__name__)

_ce: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _ce
    if _ce is None:
        _ce = CrossEncoder(
            settings.rerank_model,
            device=settings.rerank_device,
            max_length=512,
        )
    return _ce


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalize_rerank_text(text: str, *, max_chars: int = 1800) -> str:
    clean = _norm_space(str(text or ""))
    clean = re.sub(r"\[(DOC_TYPE|OFFICIAL_TITLE|ISSUING_AGENCY|LEAD_BLOCK|PATH|NODE_TYPE|CHUNK)\]\s*", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _truncate_question_words(question: str, *, max_words: int = 30) -> str:
    words = _norm_space(question).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()


def _contains_legal_header(raw: str) -> bool:
    lowered = str(raw or "").lower()
    return "[vị trí]" in lowered or "điều " in lowered or "khoản " in lowered or "điểm " in lowered


def _ensure_priority_headers(passage: Dict[str, Any], raw: str) -> str:
    text = str(raw or "")
    md = dict(passage.get("metadata") or {})
    official_title = _norm_space(
        str(md.get("official_title") or md.get("law_name") or passage.get("doc_key") or "")
    )
    header = _norm_space(
        " | ".join(
            part for part in [
                str(md.get("path_title") or ""),
                str(md.get("heading_title") or ""),
                str(md.get("article") or ""),
                str(md.get("clause") or ""),
                str(md.get("point") or ""),
            ] if _norm_space(part)
        )
    )

    lowered = text.lower()
    enriched: List[str] = []
    if official_title and "[văn bản]" not in lowered and official_title.lower() not in lowered:
        enriched.append(f"[Văn bản] {official_title}")
    if header and not _contains_legal_header(text):
        enriched.append(f"[Vị trí] {header}")
    enriched.append(text)
    return "\n\n".join(part for part in enriched if _norm_space(part))


def _build_rerank_text(passage: Dict[str, Any]) -> str:
    raw = str(
        passage.get("rerank_text_short")
        or passage.get("rerank_text")
        or passage.get("retrieval_text")
        or passage.get("text")
        or passage.get("snippet")
        or ""
    )
    enriched_raw = _ensure_priority_headers(passage, raw)
    return _normalize_rerank_text(enriched_raw)


def _intent_adjustment(question: str, passage: Dict[str, Any], query_profile: Optional[Dict[str, Any]]) -> float:
    profile = query_profile or {}
    route = str(profile.get("route") or "")
    heading_text = _norm_space(str(passage.get("heading_text") or (passage.get("metadata") or {}).get("path_title") or "")).lower()
    artifact_type = str((passage.get("metadata") or {}).get("artifact_type") or "")

    boost = 0.0
    primary_heading_term = str(profile.get("primary_heading_term") or "").strip().lower()
    conflicting = [str(x).lower() for x in (profile.get("conflicting_heading_terms") or [])]

    if route == "heading_list":
        if artifact_type == "article_bundle":
            boost += 0.12
        if primary_heading_term and primary_heading_term in heading_text:
            boost += 0.18
        if conflicting and any(term in heading_text for term in conflicting) and primary_heading_term not in heading_text:
            boost -= 0.20

    if route == "version_change":
        q = question.lower()
        if any(term in q for term in ["bãi bỏ", "hết hiệu lực"]) and any(term in heading_text for term in ["bãi bỏ", "hết hiệu lực"]):
            boost += 0.18
        if any(term in q for term in ["sửa đổi", "bổ sung", "thay thế"]) and any(term in heading_text for term in ["sửa đổi", "bổ sung", "thay thế"]):
            boost += 0.14

    return boost


def cross_rerank(
    question: str,
    passages: List[Dict[str, Any]],
    top_n: int,
    *,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not passages:
        return []

    top_n = max(0, min(int(top_n or 0), len(passages)))
    if top_n == 0:
        return []

    try:
        ce = get_cross_encoder()
        q_for_rerank = _truncate_question_words(question, max_words=30)
        pairs = [[q_for_rerank, _build_rerank_text(p)] for p in passages]
        scores = ce.predict(pairs)
    except Exception as exc:
        logger.warning("Cross rerank failed, fallback to input order: %s", exc)
        out: List[Dict[str, Any]] = []
        for p in passages[:top_n]:
            item = dict(p)
            item.setdefault("cross_score", 0.0)
            out.append(item)
        return out

    scored: List[Dict[str, Any]] = []
    for score, passage in zip(scores, passages):
        item = dict(passage)
        base = float(score)
        adjust = _intent_adjustment(question, passage, query_profile)
        item["cross_score_raw"] = base
        item["cross_score"] = base + adjust
        scored.append(item)

    scored.sort(
        key=lambda x: (
            float(x.get("cross_score", 0.0)),
            float(x.get("hybrid_score", 0.0)),
            float(x.get("doc_score", 0.0)),
        ),
        reverse=True,
    )
    return scored[:top_n]
