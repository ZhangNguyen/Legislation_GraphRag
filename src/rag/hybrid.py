from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Unicode-aware tokenizer for Vietnamese legal text.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _legal_reference_tokens(text: str) -> List[str]:
    raw = _norm_space((text or "").lower())
    extra: List[str] = []

    for m in ARTICLE_RE.finditer(raw):
        num = m.group(1)
        extra.extend([f"điều_{num}", f"article_{num}"])

    for m in CLAUSE_RE.finditer(raw):
        num = m.group(1)
        extra.extend([f"khoản_{num}", f"clause_{num}"])

    for m in POINT_RE.finditer(raw):
        ch = m.group(1).lower()
        extra.extend([f"điểm_{ch}", f"point_{ch}"])

    return extra


def tokenize(text: str) -> List[str]:
    raw = _norm_space(text).lower()
    base = [tok for tok in TOKEN_RE.findall(raw) if tok.strip()]
    return base + _legal_reference_tokens(raw)


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    *,
    idf_map: Optional[Dict[str, float]] = None,
    avgdl: float = 0.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    tf: Dict[str, int] = {}
    for tok in doc_tokens:
        tf[tok] = tf.get(tok, 0) + 1

    doc_len = max(len(doc_tokens), 1)
    avgdl_eff = max(float(avgdl or 0.0), 1.0)
    score = 0.0

    for q in query_tokens:
        freq = tf.get(q, 0)
        if freq <= 0:
            continue

        idf = float((idf_map or {}).get(q, 1.0))
        denom = freq + k1 * (1.0 - b + b * (doc_len / avgdl_eff))
        if denom <= 0:
            continue
        score += idf * (freq * (k1 + 1.0)) / denom

    return float(score)


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_rank(
    question: str,
    candidates: List[Dict[str, Any]],
    *,
    dense_key: str = "dense_score",
    text_key: str = "retrieval_text",
    alpha: float = 0.55,
    idf_map: Optional[Dict[str, float]] = None,
    avgdl: float = 0.0,
) -> List[Tuple[float, Dict[str, Any]]]:
    if not candidates:
        return []

    q_tokens = tokenize(question)

    dense_values: List[float] = []
    bm25_values: List[float] = []
    enriched: List[Dict[str, Any]] = []

    for cand in candidates:
        item = dict(cand)
        dense = float(item.get(dense_key, 0.0) or 0.0)
        text = str(item.get(text_key) or item.get("text") or item.get("snippet") or "")
        tokens = tokenize(text)
        lex = bm25_score(
            q_tokens,
            tokens,
            idf_map=idf_map,
            avgdl=avgdl,
        )
        item["bm25_score"] = float(lex)
        dense_values.append(dense)
        bm25_values.append(float(lex))
        enriched.append(item)

    dense_norm = _minmax(dense_values)
    bm25_norm = _minmax(bm25_values)

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for idx, item in enumerate(enriched):
        dense_n = dense_norm[idx] if idx < len(dense_norm) else 0.0
        bm25_n = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
        score = alpha * dense_n + (1.0 - alpha) * bm25_n
        item["dense_norm"] = float(dense_n)
        item["bm25_norm"] = float(bm25_n)
        ranked.append((float(score), item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked
