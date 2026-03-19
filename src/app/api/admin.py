from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.app.runtime import build_runtime_graph, ensure_runtime_graph, get_runtime_status
from src.rag.hierachical_summary import build_hierarchical_summaries
from src.rag.legal_versioning import summarize_versioning

router = APIRouter(prefix="/admin", tags=["admin"])


class SummaryQuestionsRequest(BaseModel):
    document_title: Optional[str] = None
    n_questions: int = 6
    include_clause: bool = False
    include_article: bool = True
    include_change: bool = True
    include_community: bool = False


@router.get("/status")
def status() -> Dict[str, Any]:
    return get_runtime_status()


@router.post("/reload-graph")
def reload_graph(
    input_dir: Optional[str] = Query(default=None),
    glob_pattern: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    graph = build_runtime_graph(input_dir=input_dir, glob_pattern=glob_pattern)
    return {
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "status": get_runtime_status(),
    }


@router.get("/versioning")
def versioning_preview(limit: int = 20) -> Dict[str, Any]:
    graph = ensure_runtime_graph()
    rows = summarize_versioning(graph)
    return {
        "count": len(rows),
        "items": rows[:limit],
    }


@router.post("/summary-questions")
def summary_questions(req: SummaryQuestionsRequest) -> Dict[str, Any]:
    graph = ensure_runtime_graph()
    graph_copy = {
        "nodes": copy.deepcopy(graph.get("nodes", [])),
        "edges": copy.deepcopy(graph.get("edges", [])),
    }

    out = build_hierarchical_summaries(
        graph=graph_copy,
        document_title=req.document_title,
        include_clause=req.include_clause,
        include_article=req.include_article,
        include_change=req.include_change,
        include_community=req.include_community,
        generate_questions=True,
        n_questions=req.n_questions,
    )

    return {
        "document_summary": out.get("document_summary"),
        "generated_questions": out.get("generated_questions", []),
        "summary_node_count": len(out.get("summary_nodes", [])),
    }
