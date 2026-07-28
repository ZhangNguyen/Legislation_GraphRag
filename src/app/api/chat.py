from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.app.question_debug_logger import new_question_log_id, write_question_debug_log
from src.app.settings import settings
from src.app.runtime import ensure_runtime_graph
from src.rag.answer_orchestrator import answer_tree_guided_with_retry
from src.rag.context_grader import grade_context
from src.rag.qa_engine import answer_with_rag
from src.rag.post_answer_verifier import repair_answer, verify_answer
from src.rag.retrieval_pipeline import retrieve_with_graph
from src.rag.retrieval_pipeline_simple import retrieve_with_graph_simple
from src.rag.tree_guided_pipeline import retrieve_tree_guided

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequestModel(BaseModel):
    question: str = Field(..., min_length=1)
    filters: Dict[str, Any] = Field(default_factory=dict)
    qdrant_top_k: Optional[int] = None
    final_top_k: Optional[int] = None
    cross_top_k: Optional[int] = None
    graph_hops: int = 2
    max_graph_nodes: int = 80
    include_debug: bool = True


@router.post("")
def chat(req: ChatRequestModel) -> Dict[str, Any]:
    request_id = new_question_log_id()
    logger.info(
        "Chat request received: id=%s chars=%s include_debug=%s",
        request_id,
        len(req.question or ""),
        req.include_debug,
    )
    request_payload = req.model_dump()
    request_payload["request_id"] = request_id
    retrieval_result: Dict[str, Any] = {}
    verification: Dict[str, Any] = {}
    adequacy: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    timings_ms: Dict[str, Any] = {}
    stage = "start"
    t0 = time.perf_counter()
    t1 = t0
    t2 = t0

    try:
        stage = "graph_load"
        graph = ensure_runtime_graph()
        t1 = time.perf_counter()

        stage = "retrieval"
        pipeline = str(getattr(settings, "retrieval_pipeline", "tree_guided") or "").lower()
        response = None
        if pipeline in {"tree", "tree_guided", "tree-guided"}:
            retrieval_result, response, adequacy = answer_tree_guided_with_retry(
                question=req.question,
                graph=graph,
                filters=req.filters,
                final_top_k=req.final_top_k,
            )
        elif pipeline in {"simple", "simple_rrf"}:
            retrieval_result = retrieve_with_graph_simple(
                question=req.question,
                graph=graph,
                filters=req.filters,
                final_top_k=req.final_top_k,
            )
            context_grade = grade_context(
                req.question,
                dict(retrieval_result.get("query_profile") or {}),
                list(retrieval_result.get("passages") or []),
            )
            retrieval_result["context_grade"] = context_grade
            retrieval_result["insufficient_context"] = context_grade.get("status") == "insufficient"
        else:
            retrieval_result = retrieve_with_graph(
                question=req.question,
                graph=graph,
                filters=req.filters,
                qdrant_top_k=req.qdrant_top_k,
                graph_hops=req.graph_hops,
                max_graph_nodes=req.max_graph_nodes,
                final_top_k=req.final_top_k,
                cross_top_k=req.cross_top_k,
            )
        t2 = time.perf_counter()

        stage = "answer"
        if response is None:
            response = answer_with_rag(
                question=req.question,
                retrieval_result=retrieval_result,
            )
        answer_text = response.answer
        if bool(getattr(settings, "enable_post_answer_verifier", True)):
            stage = "post_answer_verifier"
            verification = verify_answer(
                question=req.question,
                answer=answer_text,
                passages=list(retrieval_result.get("passages") or []),
                enable_llm=True,
            )
            if not verification.get("valid"):
                stage = "post_answer_repair"
                answer_text, verification = repair_answer(
                    question=req.question,
                    answer=answer_text,
                    passages=list(retrieval_result.get("passages") or []),
                    verification=verification,
                )
        t3 = time.perf_counter()
        timings_ms = {
            "graph_load": round((t1 - t0) * 1000, 2),
            "retrieval": round((t2 - t1) * 1000, 2),
            "answer": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }

        result = {
            "request_id": request_id,
            "answer": answer_text,
            "sources": [s.model_dump() for s in response.sources],
            "timings_ms": timings_ms,
        }

        if req.include_debug:
            result["debug"] = {
                "request_id": request_id,
                "debug_log_path": settings.question_debug_log_path,
                "mode": retrieval_result.get("mode"),
                "query_profile": retrieval_result.get("query_profile", {}),
                "reference_threshold": retrieval_result.get("reference_threshold"),
                "max_reference_hybrid": retrieval_result.get("max_reference_hybrid"),
                "rerank_veto": retrieval_result.get("rerank_veto"),
                "veto_reason": retrieval_result.get("veto_reason"),
                "topic_phrases": retrieval_result.get("topic_phrases", []),
                "top_docs": retrieval_result.get("top_docs", []),
                "candidate_pool": retrieval_result.get("candidate_pool", []),
                "passages_count": len(retrieval_result.get("passages", []) or []),
                "insufficient_context": retrieval_result.get("insufficient_context", False),
                "context_grade": retrieval_result.get("context_grade", {}),
                "filter_info": retrieval_result.get("filter_info", {}),
                "summary_candidates": retrieval_result.get("summary_candidates", []),
                "navigation_trace": retrieval_result.get("navigation_trace", []),
                "debug_flow": retrieval_result.get("debug_flow", {}),
                "retrieval_timings_ms": retrieval_result.get("retrieval_timings_ms", {}),
                "local_adequacy_judgement": adequacy or retrieval_result.get("local_adequacy_judgement", {}),
                "answer_retry_attempts": retrieval_result.get("answer_retry_attempts", []),
                "post_answer_verification": verification,
            }

        write_question_debug_log(
            request_id=request_id,
            request=request_payload,
            stage="success",
            timings_ms=timings_ms,
            retrieval_result=retrieval_result,
            response_payload=result,
            verification=verification,
        )
        return result
    except Exception as exc:
        now = time.perf_counter()
        timings_ms = {
            "graph_load": round((t1 - t0) * 1000, 2) if t1 else 0,
            "retrieval": round((t2 - t1) * 1000, 2) if t2 and t1 else 0,
            "total_until_error": round((now - t0) * 1000, 2),
        }
        write_question_debug_log(
            request_id=request_id,
            request=request_payload,
            stage=f"error:{stage}",
            timings_ms=timings_ms,
            retrieval_result=retrieval_result,
            response_payload=result,
            verification=verification,
            error=exc,
        )
        raise
