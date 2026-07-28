from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from src.app.settings import settings
from src.rag.local_adequacy_judge import judge_answer_adequacy
from src.rag.qa_engine import answer_with_rag
from src.rag.schemas import ChatResponse
from src.rag.tree_guided_pipeline import retrieve_tree_guided


def answer_tree_guided_with_retry(
    *,
    question: str,
    graph: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    final_top_k: Optional[int] = None,
    retriever: Optional[Callable[[int], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], ChatResponse, Dict[str, Any]]:
    base_top_k = int(final_top_k or getattr(settings, "tree_summary_top_k", 12) or 12)
    increment = max(1, int(getattr(settings, "tree_answer_retry_increment", 4) or 4))
    max_rounds = max(0, int(getattr(settings, "tree_answer_retry_rounds", 1) or 1))
    attempts = []

    retrieval_result: Dict[str, Any] = {}
    response: ChatResponse
    adequacy: Dict[str, Any] = {}

    for round_idx in range(max_rounds + 1):
        top_k = base_top_k + round_idx * increment
        if retriever:
            retrieval_result = retriever(top_k)
        else:
            retrieval_result = retrieve_tree_guided(
                question=question,
                graph=graph,
                filters=filters,
                final_top_k=top_k,
            )
        response = answer_with_rag(question, retrieval_result)
        adequacy = judge_answer_adequacy(question, response.answer, list(retrieval_result.get("passages") or []))
        attempts.append(
            {
                "round": round_idx,
                "top_k": top_k,
                "adequacy": adequacy,
                "passages_count": len(retrieval_result.get("passages") or []),
            }
        )
        if bool(adequacy.get("adequate")):
            break

    retrieval_result["answer_retry_attempts"] = attempts
    retrieval_result["local_adequacy_judgement"] = adequacy
    if retrieval_result.get("debug_flow") is not None:
        debug_flow = dict(retrieval_result.get("debug_flow") or {})
        debug_flow["answer_retry_attempts"] = attempts
        debug_flow["local_adequacy_judgement"] = adequacy
        retrieval_result["debug_flow"] = debug_flow
    return retrieval_result, response, adequacy
