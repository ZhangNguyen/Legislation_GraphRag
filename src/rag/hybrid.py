from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

TOKEN_RE = re.compile("r[A-Za-z0-9]+")

def tokenize(s:str)->List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def bm25_score(query_tokens:List[str], doc_tokens: List[str], k1:float=2.0,b:float=0.75)->float:
    if not query_tokens or not doc_tokens:
        return 0.0

    tf:Dict[str,int]={}
    for t in doc_tokens:
        tf[t] = tf.get(t,0)+1

    score=0.0
    doc_len=len(doc_tokens)
    avgdl = max(doc_len,1)
    for q in query_tokens:
        f = tf.get(q,0)
        if f == 0:
            continue
        idf = 1.0
        denom = f + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * (f * (k1 + 1) / denom)
    return float(score)

# Min, max scaling of scores to [0,1]
def normalize_scores(xs: List[float]) -> List[float]:
    if not xs:
        return []
    mn,mx = min(xs), max(xs)
    if mx-mn<1e-9:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def hybrid_rank(question:str, candidates: List[Dict[str,Any]], dense_key: str="dense_score",
                text_key: str="text",alpha: float=0.65,) -> List[Tuple[float,Dict[str,Any]]]:

    q_toks = tokenize(question)
    dense_scores = [float(c.get(dense_key,0.0)) for c in candidates]
    dense_norm = normalize_scores(dense_scores)
    lex_scores=[]
    for c in candidates:
        doc_toks = tokenize(c.get(text_key) or c.get("snippet") or "")
        lex_scores.append(bm25_score(q_toks,doc_toks))
    lex_norm = normalize_scores(lex_scores)
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for i,c in enumerate(candidates):
        score = alpha*dense_norm[i] + (1-alpha)*lex_norm[i]
        ranked.append((score,c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked

