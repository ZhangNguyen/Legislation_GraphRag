from __future__ import annotations

import logging
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


def _build_rerank_text(passage: Dict[str, Any]) -> str:
    return str(
        passage.get("rerank_text")
        or passage.get("retrieval_text")
        or passage.get("text")
        or passage.get("snippet")
        or ""
    ).strip()


def cross_rerank(question: str, passages: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not passages:
        return []

    top_n = max(0, min(int(top_n or 0), len(passages)))
    if top_n == 0:
        return []

    try:
        ce = get_cross_encoder()
        pairs = [[question, _build_rerank_text(p)] for p in passages]
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
        item["cross_score"] = float(score)
        scored.append(item)

    scored.sort(key=lambda x: float(x.get("cross_score", 0.0)), reverse=True)
    return scored[:top_n]
