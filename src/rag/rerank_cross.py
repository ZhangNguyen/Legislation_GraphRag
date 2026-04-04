from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

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


def _normalize_rerank_text(text: str, *, max_chars: int = 1500) -> str:
    clean = _norm_space(str(text or ""))
    clean = re.sub(
        r"\[(DOC_TYPE|OFFICIAL_TITLE|ISSUING_AGENCY|LEAD_BLOCK|PATH|NODE_TYPE|CHUNK)\]\s*",
        " ",
        clean,
    )
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def build_anchor_rerank_text(candidate: Dict[str, Any]) -> str:
    raw = str(
        candidate.get("rerank_text_short")
        or candidate.get("rerank_text")
        or candidate.get("retrieval_text")
        or candidate.get("text")
        or candidate.get("snippet")
        or ""
    )
    return _normalize_rerank_text(raw)


def cross_rerank(question: str, passages: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not passages:
        return []

    question = _norm_space(question)
    ce = get_cross_encoder()

    pairs = []
    for p in passages:
        rerank_text = build_anchor_rerank_text(p)
        pairs.append((question, rerank_text))

    scores = ce.predict(pairs)

    rescored: List[Dict[str, Any]] = []
    for p, score in zip(passages, scores):
        item = dict(p)
        item["cross_score"] = float(score)
        rescored.append(item)

    rescored.sort(key=lambda x: float(x.get("cross_score", 0.0)), reverse=True)
    return rescored[: max(1, int(top_n or 1))]