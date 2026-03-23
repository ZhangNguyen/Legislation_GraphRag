from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app.settings import settings
from src.rag.chunking_legal import legal_chunk
from src.rag.document_header import extract_document_header
from src.rag.ingestion import upsert_chunks
from src.rag.openai_clients import get_embedings
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document

logger = logging.getLogger(__name__)

_RUNTIME: Dict[str, Any] = {
    "graph": None,
    "graph_loaded_at": None,
    "graph_doc_count": 0,
    "last_graph_build_seconds": None,
    "last_reindex_seconds": None,
    "graph_build_in_progress": False,
}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "doc"


def _snapshot_path() -> Path:
    p = Path(settings.graph_snapshot_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _build_sibling_map(parent_to_children: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sibling_map: Dict[str, List[str]] = {}
    for _, children in parent_to_children.items():
        for child in children:
            sibling_map[child] = [x for x in children if x != child]
    return sibling_map


def _save_graph_snapshot(graph: Dict[str, Any]) -> None:
    p = _snapshot_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "doc_index": graph.get("doc_index", {}),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_graph_snapshot() -> Optional[Dict[str, Any]]:
    p = _snapshot_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    nodes = list(raw.get("nodes", []))
    edges = list(raw.get("edges", []))
    doc_index = dict(raw.get("doc_index", {}))
    if not nodes:
        return None
    node_index = {str(n["node_id"]): n for n in nodes if n.get("node_id")}
    parent_to_children: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        md = node.get("metadata", {}) or {}
        parent_id = md.get("parent_id")
        if parent_id:
            parent_to_children[str(parent_id)].append(str(node["node_id"]))
    sibling_map = _build_sibling_map(parent_to_children)
    for node_id, node in node_index.items():
        md = node.setdefault("metadata", {})
        md["children_ids"] = parent_to_children.get(node_id, [])
        md["sibling_ids"] = sibling_map.get(node_id, [])
    return {
        "nodes": nodes,
        "edges": edges,
        "node_index": node_index,
        "doc_index": doc_index,
        "parent_to_children": dict(parent_to_children),
        "sibling_map": sibling_map,
    }


def _apply_runtime_graph(graph: Dict[str, Any], built_seconds: float) -> Dict[str, Any]:
    _RUNTIME["graph"] = graph
    _RUNTIME["graph_loaded_at"] = time.time()
    _RUNTIME["graph_doc_count"] = len(graph.get("doc_index", {}))
    _RUNTIME["last_graph_build_seconds"] = built_seconds
    return graph


def preload_runtime_graph_from_snapshot() -> bool:
    if _RUNTIME["graph"] is not None:
        return True
    graph = _load_graph_snapshot()
    if graph is None:
        return False
    _apply_runtime_graph(graph, built_seconds=0.0)
    return True


def build_runtime_graph(
    input_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    *,
    max_workers: int = 1,
) -> Dict[str, Any]:
    _ = max_workers
    start = time.perf_counter()
    _RUNTIME["graph_build_in_progress"] = True
    input_dir = input_dir or settings.normalized_dir
    glob_pattern = glob_pattern or settings.normalized_glob

    base = Path(input_dir)
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    try:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_index: Dict[str, Dict[str, Any]] = {}
        doc_index: Dict[str, Dict[str, Any]] = {}
        parent_to_children: Dict[str, List[str]] = defaultdict(list)

        for fp in sorted(base.glob(glob_pattern)):
            if fp.suffix.lower() not in {".pdf", ".txt"}:
                continue
            text = load_document(fp)
            header = extract_document_header(text, fallback_name=fp.stem)
            doc_index[header.doc_id] = header.to_metadata()
            doc_index[header.doc_id]["file_name"] = fp.name
            doc_index[header.doc_id]["source_path"] = str(fp)

            chunks = legal_chunk(text, fallback_doc_name=fp.stem)
            for ch in chunks:
                md = dict(ch.get("metadata") or {})
                node_id = str(md.get("node_id") or md.get("chunk_id"))
                parent_id = md.get("parent_id")
                node = {
                    "node_id": node_id,
                    "node_type": md.get("node_type") or "text",
                    "text": str(ch.get("text") or "").strip(),
                    "retrieval_text": str(ch.get("retrieval_text") or "").strip(),
                    "rerank_text": str(ch.get("rerank_text") or ch.get("retrieval_text") or "").strip(),
                    "metadata": md,
                }
                nodes.append(node)
                node_index[node_id] = node
                if parent_id:
                    edges.append({"source_id": parent_id, "target_id": node_id, "relation_type": "HAS_CHILD"})
                    parent_to_children[str(parent_id)].append(node_id)

        sibling_map = _build_sibling_map(parent_to_children)
        for node_id, node in node_index.items():
            md = node.setdefault("metadata", {})
            md["children_ids"] = parent_to_children.get(node_id, md.get("children_ids", []))
            md["sibling_ids"] = sibling_map.get(node_id, md.get("sibling_ids", []))
            if md.get("doc_id") and md["doc_id"] in doc_index:
                for key in [
                    "official_title",
                    "doc_type",
                    "law_name",
                    "law_type",
                    "source",
                    "year",
                    "lead_block",
                    "doc_number",
                    "issuing_agency",
                ]:
                    if md.get(key) in (None, ""):
                        val = doc_index[md["doc_id"]].get(key)
                        if val not in (None, ""):
                            md[key] = val

        graph = {
            "nodes": nodes,
            "edges": edges,
            "node_index": node_index,
            "doc_index": doc_index,
            "parent_to_children": dict(parent_to_children),
            "sibling_map": sibling_map,
        }
        _apply_runtime_graph(graph, built_seconds=time.perf_counter() - start)
        _save_graph_snapshot(graph)
        return graph
    finally:
        _RUNTIME["graph_build_in_progress"] = False


def ensure_runtime_graph() -> Dict[str, Any]:
    if _RUNTIME["graph"] is not None:
        return _RUNTIME["graph"]
    if preload_runtime_graph_from_snapshot():
        return _RUNTIME["graph"]
    return build_runtime_graph()


def get_runtime_status() -> Dict[str, Any]:
    graph = _RUNTIME["graph"] or {}
    return {
        "graph_loaded": _RUNTIME["graph"] is not None,
        "graph_loaded_at": _RUNTIME["graph_loaded_at"],
        "graph_doc_count": _RUNTIME["graph_doc_count"],
        "graph_node_count": len(graph.get("nodes", [])),
        "graph_edge_count": len(graph.get("edges", [])),
        "last_graph_build_seconds": _RUNTIME["last_graph_build_seconds"],
        "last_reindex_seconds": _RUNTIME["last_reindex_seconds"],
        "graph_build_in_progress": _RUNTIME["graph_build_in_progress"],
        "graph_snapshot_path": str(_snapshot_path()),
        "graph_snapshot_exists": _snapshot_path().exists(),
    }


def save_current_runtime_graph_snapshot(force_build: bool = False) -> Dict[str, Any]:
    graph = _RUNTIME.get("graph")
    if graph is None:
        if not force_build:
            raise RuntimeError("Runtime graph is empty. Build/load graph first or use force_build=true.")
        graph = build_runtime_graph()
    _save_graph_snapshot(graph)
    return {
        "snapshot_path": str(_snapshot_path()),
        "snapshot_exists": _snapshot_path().exists(),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
    }


def reindex_qdrant_from_normalized(
    input_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    *,
    semantic_merge: bool = False,
    min_chars: int = 350,
    sim_threshold: float = 0.88,
    max_merged_chars: int = 1600,
    max_chars: int = 1600,
    overlap: int = 120,
) -> Dict[str, Any]:
    _ = (semantic_merge, min_chars, sim_threshold, max_merged_chars, max_chars, overlap)
    start = time.perf_counter()
    input_dir = input_dir or settings.normalized_dir
    glob_pattern = glob_pattern or settings.normalized_glob

    base = Path(input_dir)
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    embeddings = get_embedings()
    client = get_qdrant_client()
    ensure_collection(client)

    total_chunks = 0
    total_docs = 0
    for fp in sorted(base.glob(glob_pattern)):
        if fp.suffix.lower() not in {".pdf", ".txt"}:
            continue
        text = load_document(fp)
        header = extract_document_header(text, fallback_name=fp.stem)
        chunks = legal_chunk(text, fallback_doc_name=fp.stem)
        total_chunks += int(upsert_chunks(chunks, embeddings=embeddings, meta=header.to_metadata()))
        total_docs += 1

    _RUNTIME["last_reindex_seconds"] = time.perf_counter() - start
    return {
        "docs_indexed": total_docs,
        "chunks_indexed": total_chunks,
        "seconds": _RUNTIME["last_reindex_seconds"],
    }


def rebuild_everything(input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    qdrant_info = reindex_qdrant_from_normalized(input_dir=input_dir, glob_pattern=glob_pattern)
    graph = build_runtime_graph(input_dir=input_dir, glob_pattern=glob_pattern)
    return {
        "qdrant": qdrant_info,
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "docs": _RUNTIME["graph_doc_count"],
    }
