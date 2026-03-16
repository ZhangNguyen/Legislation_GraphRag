from __future__ import annotations
from typing import Any, Dict, List

# pip install sentence-transformers
from sentence_transformers import CrossEncoder

from src.app.settings import settings

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

def cross_rerank(question: str, passages: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not passages:
        return []
    if len(passages) <= top_n:
        return passages

    ce = get_cross_encoder()
    pairs = []
    for p in passages:
        text = p.get("text") or p.get("snippet") or ""
        pairs.append([question, text])

    scores = ce.predict(pairs)  # higher = more relevant
    scored = list(zip(scores, passages))
    scored.sort(key=lambda x: float(x[0]), reverse=True)
    return [p for _, p in scored[:top_n]]
