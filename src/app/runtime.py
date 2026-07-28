from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

from src.rag.graph_builder import (
    build_runtime_graph as _build_runtime_graph,
    load_graph_snapshot,
    rebuild_everything as _rebuild_everything,
    reindex_qdrant_from_normalized as _reindex_qdrant_from_normalized,
    save_graph_snapshot,
)

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = Lock()
_RUNTIME: Dict[str, Any] = {
    "graph": None,
    "graph_loaded_at": None,
    "graph_doc_count": 0,
    "last_graph_build_seconds": None,
    "last_snapshot_load_seconds": None,
    "last_snapshot_save_seconds": None,
    "last_reindex_seconds": None,
    "startup_preload_completed": False,
}


def _graph_doc_count(graph: Optional[Dict[str, Any]]) -> int:
    if not graph:
        return 0
    doc_index = graph.get("doc_index", {}) or {}
    return len(doc_index)


def _apply_runtime_graph(graph: Optional[Dict[str, Any]], *, load_seconds: float = 0.0, build_seconds: Optional[float] = None) -> Optional[Dict[str, Any]]:
    _RUNTIME["graph"] = graph
    _RUNTIME["graph_loaded_at"] = time.time() if graph else None
    _RUNTIME["graph_doc_count"] = _graph_doc_count(graph)
    _RUNTIME["last_snapshot_load_seconds"] = load_seconds
    if build_seconds is not None:
        _RUNTIME["last_graph_build_seconds"] = build_seconds
    return graph


def clear_runtime_graph() -> Dict[str, Any]:
    with _RUNTIME_LOCK:
        _RUNTIME["graph"] = None
        _RUNTIME["graph_loaded_at"] = None
        _RUNTIME["graph_doc_count"] = 0
        _RUNTIME["last_graph_build_seconds"] = None
        _RUNTIME["last_snapshot_load_seconds"] = None
        _RUNTIME["last_snapshot_save_seconds"] = None
        _RUNTIME["startup_preload_completed"] = False
        return {
            "cleared": True,
            "graph_loaded": False,
        }


def preload_runtime_graph_from_snapshot() -> bool:
    t0 = time.perf_counter()
    graph = load_graph_snapshot()
    elapsed = time.perf_counter() - t0

    if not graph:
        logger.info("No snapshot loaded.")
        return False

    _apply_runtime_graph(graph, load_seconds=elapsed)
    logger.info("Runtime graph loaded from snapshot in %.2fs with %s docs.", elapsed, _RUNTIME["graph_doc_count"])
    return True


def build_runtime_graph() -> Dict[str, Any]:
    t0 = time.perf_counter()
    built_graph = _build_runtime_graph()
    build_elapsed = time.perf_counter() - t0

    save_t0 = time.perf_counter()
    save_graph_snapshot(built_graph)
    save_elapsed = time.perf_counter() - save_t0
    _RUNTIME["last_snapshot_save_seconds"] = save_elapsed

    load_t0 = time.perf_counter()
    loaded_graph = load_graph_snapshot()
    load_elapsed = time.perf_counter() - load_t0

    if not loaded_graph:
        # fallback rất hiếm: save xong mà load lại lỗi thì vẫn giữ graph vừa build
        logger.warning("Snapshot save succeeded but reload failed. Falling back to in-memory built graph.")
        _apply_runtime_graph(built_graph, load_seconds=0.0, build_seconds=build_elapsed)
        return built_graph

    _apply_runtime_graph(loaded_graph, load_seconds=load_elapsed, build_seconds=build_elapsed)
    logger.info(
        "Runtime graph built in %.2fs, snapshot saved in %.2fs, reloaded from snapshot in %.2fs, docs=%s.",
        build_elapsed,
        save_elapsed,
        load_elapsed,
        _RUNTIME["graph_doc_count"],
    )
    return loaded_graph


def ensure_runtime_graph() -> Dict[str, Any]:
    graph = _RUNTIME.get("graph")
    if graph is not None:
        return graph

    with _RUNTIME_LOCK:
        graph = _RUNTIME.get("graph")
        if graph is not None:
            return graph

        if preload_runtime_graph_from_snapshot():
            return _RUNTIME["graph"]

        return build_runtime_graph()


def preload_runtime_on_startup(
    *,
    preload_graph_snapshot: bool = True,
    build_graph_if_missing: bool = True,
    warm_services: bool = True,
) -> Dict[str, Any]:
    del warm_services

    with _RUNTIME_LOCK:
        if preload_graph_snapshot:
            loaded = preload_runtime_graph_from_snapshot()
            if loaded:
                _RUNTIME["startup_preload_completed"] = True
                return get_runtime_status()

        if build_graph_if_missing:
            build_runtime_graph()

        _RUNTIME["startup_preload_completed"] = True
        return get_runtime_status()


def rebuild_everything(*, input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = _rebuild_everything(input_dir=input_dir, glob_pattern=glob_pattern)
    elapsed = time.perf_counter() - t0

    graph = result.get("graph")
    if graph:
        save_t0 = time.perf_counter()
        save_graph_snapshot(graph)
        _RUNTIME["last_snapshot_save_seconds"] = time.perf_counter() - save_t0

        load_t0 = time.perf_counter()
        reloaded = load_graph_snapshot()
        load_elapsed = time.perf_counter() - load_t0

        if reloaded:
            _apply_runtime_graph(reloaded, load_seconds=load_elapsed)
        else:
            _apply_runtime_graph(graph, load_seconds=0.0)

    _RUNTIME["last_reindex_seconds"] = elapsed
    return {
        "ok": True,
        "seconds": elapsed,
        "graph_loaded": _RUNTIME["graph"] is not None,
        "graph_doc_count": _RUNTIME["graph_doc_count"],
    }


def reindex_qdrant_from_normalized(*, input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = _reindex_qdrant_from_normalized(input_dir=input_dir, glob_pattern=glob_pattern)
    elapsed = time.perf_counter() - t0
    _RUNTIME["last_reindex_seconds"] = elapsed

    payload = dict(result or {})
    payload["seconds"] = elapsed
    return payload


def save_current_runtime_graph_snapshot() -> Dict[str, Any]:
    graph = _RUNTIME.get("graph")
    if not graph:
        return {
            "ok": False,
            "reason": "graph_not_loaded",
        }

    t0 = time.perf_counter()
    save_graph_snapshot(graph)
    elapsed = time.perf_counter() - t0
    _RUNTIME["last_snapshot_save_seconds"] = elapsed

    return {
        "ok": True,
        "seconds": elapsed,
        "graph_doc_count": _RUNTIME["graph_doc_count"],
    }


def get_runtime_graph() -> Optional[Dict[str, Any]]:
    return _RUNTIME.get("graph")


def get_runtime_status() -> Dict[str, Any]:
    return {
        "graph_loaded": _RUNTIME.get("graph") is not None,
        "graph_loaded_at": _RUNTIME.get("graph_loaded_at"),
        "graph_doc_count": _RUNTIME.get("graph_doc_count", 0),
        "last_graph_build_seconds": _RUNTIME.get("last_graph_build_seconds"),
        "last_snapshot_load_seconds": _RUNTIME.get("last_snapshot_load_seconds"),
        "last_snapshot_save_seconds": _RUNTIME.get("last_snapshot_save_seconds"),
        "last_reindex_seconds": _RUNTIME.get("last_reindex_seconds"),
        "startup_preload_completed": _RUNTIME.get("startup_preload_completed", False),
    }
