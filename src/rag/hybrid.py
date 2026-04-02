from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)


TOPIC_PHRASE_MIN_TOKEN_LEN = 2
STOPWORDS = {
    "và", "của", "các", "những", "trong", "trước", "sau", "cho", "về", "việc", "để", "tại", "theo",
    "là", "ở", "có", "khi", "này", "đó", "với", "một", "nhiều", "được", "đang", "cần",
}


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


def _ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def tokenize(text: str) -> List[str]:
    raw = _norm_space(text).lower()
    base = [tok for tok in TOKEN_RE.findall(raw) if tok.strip()]
    grams = _ngrams(base, 2) + _ngrams(base, 3)
    return base + grams + _legal_reference_tokens(raw)


def extract_topic_phrases(question: str) -> List[str]:
    tokens = [tok for tok in TOKEN_RE.findall(_norm_space(question).lower()) if tok.strip()]
    out: List[str] = []
    for n in (3, 2):
        for gram in _ngrams(tokens, n):
            parts = gram.split()
            if all(p in STOPWORDS for p in parts):
                continue
            if len([p for p in parts if p not in STOPWORDS]) < TOPIC_PHRASE_MIN_TOKEN_LEN:
                continue
            out.append(gram)
    seen = set()
    uniq: List[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def phrase_overlap_score(question: str, text: str) -> Tuple[float, List[str]]:
    phrases = extract_topic_phrases(question)
    hay = _norm_space(text).lower()
    hits = [p for p in phrases if p in hay]
    if not phrases:
        return 0.0, hits
    return float(len(hits) / len(phrases)), hits


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
    tf = Counter(doc_tokens)
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


def rrf_merge(rank_lists: List[List[str]], *, k: int = 60) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rank_list in rank_lists:
        for idx, item_id in enumerate(rank_list, start=1):
            out[item_id] = out.get(item_id, 0.0) + 1.0 / (k + idx)
    return out


def field_dense_bm25(question: str, texts: List[str], dense_scores: Optional[List[float]] = None) -> List[Dict[str, float]]:
    q_tokens = tokenize(question)
    bm25_values = [bm25_score(q_tokens, tokenize(text)) for text in texts]
    bm25_norm = _minmax(bm25_values)
    dense_scores = dense_scores or [0.0] * len(texts)
    dense_norm = _minmax([float(x or 0.0) for x in dense_scores])
    out: List[Dict[str, float]] = []
    for idx in range(len(texts)):
        out.append(
            {
                "dense": float(dense_scores[idx] if idx < len(dense_scores) else 0.0),
                "dense_norm": float(dense_norm[idx] if idx < len(dense_norm) else 0.0),
                "bm25": float(bm25_values[idx]),
                "bm25_norm": float(bm25_norm[idx] if idx < len(bm25_norm) else 0.0),
            }
        )
    return out
