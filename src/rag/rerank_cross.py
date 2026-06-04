from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from src.app.settings import settings

logger = logging.getLogger(__name__)
_ce: Any = None
_load_failed = False


def get_cross_encoder() -> Any:
    global _ce, _load_failed
    if not bool(getattr(settings, "enable_reranker", False)):
        return None
    if _load_failed:
        return None
    if _ce is None:
        try:
            from sentence_transformers import CrossEncoder

            _ce = CrossEncoder(
                settings.rerank_model,
                device=settings.rerank_device,
                max_length=512,
            )
        except Exception as exc:
            _load_failed = True
            logger.warning("Reranker disabled after model load failure: %s", exc)
            return None
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
    if ce is None:
        return passages[: max(1, int(top_n or len(passages)))]

    pairs = []
    input_top_k = max(1, int(getattr(settings, "rerank_input_top_k", 30) or 30))
    working = passages[:input_top_k]
    for p in working:
        rerank_text = build_anchor_rerank_text(p)
        pairs.append((question, rerank_text))

    try:
        scores = ce.predict(pairs)
    except Exception as exc:
        logger.warning("Reranker disabled for this request after predict failure: %s", exc)
        return passages[: max(1, int(top_n or len(passages)))]

    rescored: List[Dict[str, Any]] = []
    for p, score in zip(working, scores):
        item = dict(p)
        item["cross_score"] = float(score)
        rescored.append(item)

    rescored.sort(key=lambda x: float(x.get("cross_score", 0.0)), reverse=True)
    return rescored[: max(1, int(top_n or 1))]
