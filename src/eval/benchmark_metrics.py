from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from src.rag.openai_clients import get_embedings, get_llm

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(_norm_space(text)) if t.strip()}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?;])\s+", _norm_space(text))
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(union or 1)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def split_claims(answer: str) -> List[str]:
    """Tách answer thành các claim ngắn để chấm faithfulness."""
    claims = _split_sentences(answer)
    return claims or ([_norm_space(answer)] if _norm_space(answer) else [])


def score_faithfulness(
    answer: str,
    contexts: List[str],
    *,
    support_threshold: float = 0.2,
) -> Dict[str, Any]:
    claims = split_claims(answer)
    context_blob = [c for c in contexts if _norm_space(c)]

    judged: List[Dict[str, Any]] = []
    supported = 0

    for c in claims:
        best = 0.0
        best_idx = -1
        for i, ctx in enumerate(context_blob):
            s = _jaccard(c, ctx)
            if s > best:
                best = s
                best_idx = i

        is_supported = best >= support_threshold
        supported += int(is_supported)
        judged.append(
            {
                "claim": c,
                "supported": is_supported,
                "best_support_score": round(best, 4),
                "support_context_index": best_idx,
            }
        )

    score = float(supported) / float(len(claims) or 1)
    return {
        "score": round(score, 4),
        "supported_claims": supported,
        "total_claims": len(claims),
        "claims": judged,
    }


def _llm_generate_questions_from_answer(answer: str, n: int) -> List[str]:
    llm = get_llm()
    prompt = (
        "Sinh {n} câu hỏi ngắn (tiếng Việt) mà câu trả lời bên dưới có thể trả lời. "
        "Trả về mỗi câu trên một dòng, không đánh số.\n\n"
        "ANSWER:\n{ans}"
    ).format(n=n, ans=answer)

    resp = llm.invoke(prompt)
    lines = [
        _norm_space(re.sub(r"^[\-\d\.)\s]+", "", l))
        for l in str(getattr(resp, "content", "") or "").splitlines()
    ]
    lines = [l for l in lines if l]
    return lines[:n]


def _heuristic_generate_questions_from_answer(answer: str, n: int) -> List[str]:
    sents = _split_sentences(answer)
    if not sents:
        return []

    out: List[str] = []
    for s in sents[:n]:
        out.append(f"Nội dung nào được nêu trong câu: '{s}'?")
    while len(out) < n:
        out.append("Thông tin chính của câu trả lời là gì?")
    return out[:n]


def generate_questions_from_answer(answer: str, n: int = 3) -> List[str]:
    try:
        qs = _llm_generate_questions_from_answer(answer, n)
        if qs:
            return qs[:n]
    except Exception:
        pass
    return _heuristic_generate_questions_from_answer(answer, n)


def score_answer_relevancy(
    question: str,
    answer: str,
    *,
    n_generated_questions: int = 3,
) -> Dict[str, Any]:
    generated = generate_questions_from_answer(answer, n=n_generated_questions)
    if not generated:
        return {
            "score": 0.0,
            "generated_questions": [],
            "similarities": [],
        }

    sims: List[float] = []
    try:
        emb = get_embedings()
        q_vec = emb.embed_query(question)
        for gq in generated:
            gq_vec = emb.embed_query(gq)
            sims.append(_cosine(q_vec, gq_vec))
    except Exception:
        sims = [_jaccard(question, gq) for gq in generated]

    score = sum(sims) / float(len(sims) or 1)
    return {
        "score": round(score, 4),
        "generated_questions": generated,
        "similarities": [round(x, 4) for x in sims],
    }


def score_context_precision(question: str, contexts: List[str]) -> Dict[str, Any]:
    relevance = [1 if _jaccard(question, c) >= 0.08 else 0 for c in contexts]

    precisions: List[float] = []
    rel_count = 0
    for i, rel in enumerate(relevance, start=1):
        rel_count += rel
        precisions.append(rel_count / i)

    weighted = [p for p, rel in zip(precisions, relevance) if rel == 1]
    if not weighted:
        return {
            "score": 0.0,
            "relevance_labels": relevance,
            "precision_at_k": [round(x, 4) for x in precisions],
        }

    score = sum(weighted) / float(len(weighted))
    return {
        "score": round(score, 4),
        "relevance_labels": relevance,
        "precision_at_k": [round(x, 4) for x in precisions],
    }


def score_context_recall(reference_answer: str, contexts: List[str]) -> Dict[str, Any]:
    refs = _split_sentences(reference_answer)
    if not refs:
        return {
            "score": 0.0,
            "covered_sentences": 0,
            "total_sentences": 0,
        }

    covered = 0
    detail: List[Dict[str, Any]] = []
    for sent in refs:
        best = max((_jaccard(sent, c) for c in contexts), default=0.0)
        ok = best >= 0.1
        covered += int(ok)
        detail.append({"sentence": sent, "covered": ok, "best_overlap": round(best, 4)})

    score = covered / float(len(refs) or 1)
    return {
        "score": round(score, 4),
        "covered_sentences": covered,
        "total_sentences": len(refs),
        "sentences": detail,
    }


def evaluate_rag_response(
    *,
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    n_generated_questions: int = 3,
) -> Dict[str, Any]:
    faithfulness = score_faithfulness(answer, contexts)
    answer_relevancy = score_answer_relevancy(
        question,
        answer,
        n_generated_questions=n_generated_questions,
    )
    context_precision = score_context_precision(question, contexts)
    context_recall = score_context_recall(reference_answer or answer, contexts)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "meta": {
            "context_count": len(contexts),
            "uses_reference_answer": bool(reference_answer),
            "n_generated_questions": n_generated_questions,
        },
    }
