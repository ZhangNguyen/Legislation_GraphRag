from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from src.app.runtime import (
    build_runtime_graph,
    ensure_runtime_graph,
    get_runtime_status,
    save_current_runtime_graph_snapshot,
)
from src.rag.legal_versioning import summarize_versioning

router = APIRouter(prefix="/admin", tags=["admin"])


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


@router.post("/save-snapshot")
def save_snapshot(force_build: bool = False) -> Dict[str, Any]:
    result = save_current_runtime_graph_snapshot(force_build=force_build)
    result["status"] = get_runtime_status()
    return result
