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
    logger.info("Chat step graph_ready: %.2f ms", (t1 - t0) * 1000.0)

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
    logger.info(
        "Chat step retrieval_done: %.2f ms passages=%s",
        (t2 - t1) * 1000.0,
        len(retrieval_result.get("passages", []) or []),
    )

    response = answer_with_rag(
        question=req.question,
        retrieval_result=retrieval_result,
    )
    t3 = time.perf_counter()
    logger.info("Chat step answer_done: %.2f ms total=%.2f ms", (t3 - t2) * 1000.0, (t3 - t0) * 1000.0)

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
        all_passages = retrieval_result.get("passages", []) or []
        result["debug"] = {
            "filters": retrieval_result.get("filters", {}),
            "mode": retrieval_result.get("mode"),
            "qdrant_top_k": retrieval_result.get("qdrant_top_k"),
            "final_top_k": retrieval_result.get("final_top_k"),
            "cross_top_k": retrieval_result.get("cross_top_k"),
            "passages_count": len(all_passages),
            "top_passages": all_passages,
        }

    return result
