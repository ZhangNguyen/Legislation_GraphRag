from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.app.runtime import ensure_runtime_graph
from src.rag.qa_engine import answer_with_rag
from src.rag.retrieval_pipeline import retrieve_with_graph

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
    logger.info("Chat request received: chars=%s include_debug=%s", len(req.question or ""), req.include_debug)

    t0 = time.perf_counter()
    graph = ensure_runtime_graph()
    t1 = time.perf_counter()

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

    response = answer_with_rag(
        question=req.question,
        retrieval_result=retrieval_result,
    )
    t3 = time.perf_counter()

    result: Dict[str, Any] = {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "timings_ms": {
            "graph_load": round((t1 - t0) * 1000, 2),
            "retrieval": round((t2 - t1) * 1000, 2),
            "answer": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        },
    }

    if req.include_debug:
        result["debug"] = {
            "mode": retrieval_result.get("mode"),
            "reference_threshold": retrieval_result.get("reference_threshold"),
            "max_reference_hybrid": retrieval_result.get("max_reference_hybrid"),
            "rerank_veto": retrieval_result.get("rerank_veto"),
            "veto_reason": retrieval_result.get("veto_reason"),
            "topic_phrases": retrieval_result.get("topic_phrases", []),
            "top_docs": retrieval_result.get("top_docs", []),
            "candidate_pool": retrieval_result.get("candidate_pool", []),
            "passages_count": len(retrieval_result.get("passages", []) or []),
            "insufficient_context": retrieval_result.get("insufficient_context", False),
        }

    return result