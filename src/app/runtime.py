from __future__ import annotations

import copy
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app.settings import settings
from src.rag.hierachical_summary import build_hierarchical_summaries
from src.rag.ingestion import ingest_document, upsert_chunks
from src.rag.legal_versioning import annotate_graph_with_versioning
from src.rag.openai_clients import get_embedings
from src.rag.chunking_legal import legal_chunk
from src.rag.graph_builder import materialize_graph_from_ingestion
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document

try:
    from src.rag.chunking_legal import semantic_merge_safe
except Exception:
    semantic_merge_safe = None


_RUNTIME: Dict[str, Any] = {
    "graph": None,
    "graph_loaded_at": None,
    "graph_doc_count": 0,
    "last_graph_build_seconds": None,
    "last_reindex_seconds": None,
    "graph_build_in_progress": False,
}

logger = logging.getLogger(__name__)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "doc"


def _build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        src = e.get("source_id")
        if src:
            adj[str(src)].append(e)
    return dict(adj)


def _build_reverse_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rev: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        tgt = e.get("target_id")
        if tgt:
            rev[str(tgt)].append(e)
    return dict(rev)


def _dedup_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_id = n.get("node_id")
        if not node_id:
            continue
        out[str(node_id)] = n
    return list(out.values())


def _dedup_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for e in edges:
        sig = (
            str(e.get("source_id", "")).strip().lower(),
            str(e.get("target_id", "")).strip().lower(),
            str(e.get("relation_type", "")).strip().lower(),
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(e)
    return out


def _prefix_graph_ids(graph: Dict[str, Any], doc_prefix: str) -> Dict[str, Any]:
    graph = copy.deepcopy(graph)
    id_map: Dict[str, str] = {}

    for node in graph.get("nodes", []):
        old_id = str(node.get("node_id", "")).strip()
        if not old_id:
            continue
        new_id = f"{doc_prefix}::{old_id}"
        id_map[old_id] = new_id
        node["node_id"] = new_id

        md = node.get("metadata", {}) or {}
        if md.get("node_id"):
            md["node_id"] = new_id

        if isinstance(md.get("source_node_ids"), list):
            md["source_node_ids"] = [id_map.get(x, f"{doc_prefix}::{x}") for x in md["source_node_ids"]]

        if isinstance(md.get("child_summary_ids"), list):
            md["child_summary_ids"] = [id_map.get(x, f"{doc_prefix}::{x}") for x in md["child_summary_ids"]]

        node["metadata"] = md

    for edge in graph.get("edges", []):
        src = str(edge.get("source_id", "")).strip()
        tgt = str(edge.get("target_id", "")).strip()
        if src:
            edge["source_id"] = id_map.get(src, f"{doc_prefix}::{src}")
        if tgt:
            edge["target_id"] = id_map.get(tgt, f"{doc_prefix}::{tgt}")

    return graph


def _prefix_chunks(chunks: List[Dict[str, Any]], doc_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        item = dict(ch)
        md = dict(item.get("metadata", {}) or {})

        old_node_id = md.get("node_id")
        if old_node_id:
            md["node_id"] = f"{doc_prefix}::{old_node_id}"

        old_chunk_id = md.get("chunk_id")
        if old_chunk_id:
            md["chunk_id"] = f"{doc_prefix}::{old_chunk_id}"

        item["metadata"] = md
        out.append(item)
    return out


def _merge_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []

    for g in graphs:
        all_nodes.extend(g.get("nodes", []))
        all_edges.extend(g.get("edges", []))

    nodes = _dedup_nodes(all_nodes)
    edges = _dedup_edges(all_edges)

    node_index = {str(n["node_id"]): n for n in nodes if n.get("node_id")}
    adjacency = _build_adjacency(edges)
    reverse_adjacency = _build_reverse_adjacency(edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "node_index": node_index,
        "adjacency": adjacency,
        "reverse_adjacency": reverse_adjacency,
    }


def _snapshot_path() -> Path:
    p = Path(settings.graph_snapshot_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def save_graph_snapshot(graph: Dict[str, Any]) -> Path:
    p = _snapshot_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "nodes": list(graph.get("nodes", [])),
        "edges": list(graph.get("edges", [])),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Graph snapshot saved: %s", p)
    return p


def load_graph_snapshot() -> Optional[Dict[str, Any]]:
    p = _snapshot_path()
    if not p.exists():
        return None

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read graph snapshot %s: %s", p, exc)
        return None

    if not isinstance(raw, dict):
        return None

    nodes = list(raw.get("nodes", []))
    edges = list(raw.get("edges", []))
    if not nodes:
        return None

    graph = {
        "nodes": nodes,
        "edges": edges,
        "node_index": {str(n["node_id"]): n for n in nodes if n.get("node_id")},
        "adjacency": _build_adjacency(edges),
        "reverse_adjacency": _build_reverse_adjacency(edges),
    }
    logger.info("Graph snapshot loaded: %s nodes=%s edges=%s", p, len(nodes), len(edges))
    return graph


def _apply_runtime_graph(graph: Dict[str, Any], *, built_seconds: float) -> Dict[str, Any]:
    _RUNTIME["graph"] = graph
    _RUNTIME["graph_loaded_at"] = time.time()
    _RUNTIME["graph_doc_count"] = len(
        {str(n.get("metadata", {}).get("law_name", "")) for n in graph.get("nodes", []) if n.get("metadata")}
    )
    _RUNTIME["last_graph_build_seconds"] = built_seconds
    return graph


def preload_runtime_graph_from_snapshot() -> bool:
    if _RUNTIME["graph"] is not None:
        return True
    snap = load_graph_snapshot()
    if snap is None:
        return False
    _apply_runtime_graph(snap, built_seconds=0.0)
    return True


def save_current_runtime_graph_snapshot(force_build: bool = False) -> Dict[str, Any]:
    graph = _RUNTIME.get("graph")
    if graph is None:
        if not force_build:
            raise RuntimeError("Runtime graph is empty. Build/load graph first or use force_build=true.")
        graph = build_runtime_graph()

    p = save_graph_snapshot(graph)
    return {
        "snapshot_path": str(p),
        "snapshot_exists": p.exists(),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
    }


def build_runtime_graph(
    input_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    _RUNTIME["graph_build_in_progress"] = True

    input_dir = input_dir or settings.normalized_dir
    glob_pattern = glob_pattern or settings.normalized_glob
    logger.info("Graph build started: input_dir=%s glob=%s", input_dir, glob_pattern)

    base = Path(input_dir)
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    per_doc_graphs: List[Dict[str, Any]] = []
    doc_count = 0

    try:
        for fp in sorted(base.glob(glob_pattern)):
            if fp.suffix.lower() not in {".pdf", ".txt"}:
                continue

            doc_started = time.perf_counter()
            text = load_document(fp)
            doc_title = fp.stem
            doc_prefix = _slugify(doc_title)
            logger.info("Graph build doc start: %s (chars=%s)", fp.name, len(text))

            ingestion_output = ingest_document(text=text)
            logger.info(
                "Graph build ingestion done: %s nodes=%s edges=%s chunks=%s",
                fp.name,
                len(ingestion_output.get("graph_nodes", [])),
                len(ingestion_output.get("graph_edges", [])),
                len(ingestion_output.get("chunks", [])),
            )

            graph = materialize_graph_from_ingestion(ingestion_output)
            graph = build_hierarchical_summaries(
                graph=graph,
                document_title=doc_title,
                include_clause=False,
                include_article=True,
                include_change=True,
                include_community=False,
            )["graph"]

            graph = annotate_graph_with_versioning(graph)
            graph = _prefix_graph_ids(graph, doc_prefix)

            per_doc_graphs.append(graph)
            doc_count += 1
            logger.info(
                "Graph build doc done: %s graph_nodes=%s graph_edges=%s took=%.2fs",
                fp.name,
                len(graph.get("nodes", [])),
                len(graph.get("edges", [])),
                time.perf_counter() - doc_started,
            )

        merged = _merge_graphs(per_doc_graphs)

        _apply_runtime_graph(merged, built_seconds=time.perf_counter() - start)
        save_graph_snapshot(merged)
        logger.info(
            "Graph build finished: docs=%s nodes=%s edges=%s took=%.2fs",
            doc_count,
            len(merged.get("nodes", [])),
            len(merged.get("edges", [])),
            _RUNTIME["last_graph_build_seconds"],
        )
    finally:
        _RUNTIME["graph_build_in_progress"] = False

    return merged


def ensure_runtime_graph() -> Dict[str, Any]:
    if _RUNTIME["graph"] is not None:
        return _RUNTIME["graph"]
    if preload_runtime_graph_from_snapshot():
        return _RUNTIME["graph"]
    return build_runtime_graph()


def get_runtime_status() -> Dict[str, Any]:
    graph = _RUNTIME["graph"]
    return {
        "graph_loaded": graph is not None,
        "graph_loaded_at": _RUNTIME["graph_loaded_at"],
        "graph_doc_count": _RUNTIME["graph_doc_count"],
        "graph_node_count": len((graph or {}).get("nodes", [])),
        "graph_edge_count": len((graph or {}).get("edges", [])),
        "last_graph_build_seconds": _RUNTIME["last_graph_build_seconds"],
        "last_reindex_seconds": _RUNTIME["last_reindex_seconds"],
        "graph_build_in_progress": _RUNTIME["graph_build_in_progress"],
        "graph_snapshot_path": str(_snapshot_path()),
        "graph_snapshot_exists": _snapshot_path().exists(),
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

        doc_title = fp.stem
        doc_prefix = _slugify(doc_title)
        text = load_document(fp)

        chunks = legal_chunk(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            continue

        if semantic_merge:
            if semantic_merge_safe is None:
                raise RuntimeError("semantic_merge_safe not available in src.rag.chunking_legal")
            chunks = semantic_merge_safe(
                chunks,
                embeddings=embeddings,
                min_chars=min_chars,
                sim_threshold=sim_threshold,
                max_merged_chars=max_merged_chars,
            )

        chunks = _prefix_chunks(chunks, doc_prefix)

        meta = {
            "law_name": doc_title,
            "law_type": "Unknown",
            "year": 0,
            "source": "LocalFile",
        }

        n = upsert_chunks(chunks, embeddings=embeddings, meta=meta)
        total_chunks += int(n)
        total_docs += 1

    _RUNTIME["last_reindex_seconds"] = time.perf_counter() - start
    return {
        "docs_indexed": total_docs,
        "chunks_indexed": total_chunks,
        "seconds": _RUNTIME["last_reindex_seconds"],
    }


def rebuild_everything(
    input_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    qdrant_info = reindex_qdrant_from_normalized(
        input_dir=input_dir,
        glob_pattern=glob_pattern,
    )
    graph = build_runtime_graph(
        input_dir=input_dir,
        glob_pattern=glob_pattern,
    )
    return {
        "qdrant": qdrant_info,
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "docs": _RUNTIME["graph_doc_count"],
    }
