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
from src.rag.chunking_legal import legal_chunk
from src.rag.document_header import extract_document_header
from src.rag.ingestion import upsert_chunks
from src.rag.openai_clients import get_embedings
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
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
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


def _snapshot_path() -> Path:
    p = Path(settings.graph_snapshot_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _node_aliases(node: Dict[str, Any]) -> List[str]:
    md = dict(node.get("metadata") or {})
    aliases: List[str] = []
    for raw in [
        node.get("node_id"),
        md.get("node_id"),
        md.get("chunk_id"),
        md.get("source_node_id"),
    ]:
        val = str(raw or "").strip()
        if val:
            aliases.append(val)
    # Dedup while preserving order
    seen = set()
    out: List[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def _build_alias_index(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    alias_index: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        for alias in _node_aliases(node):
            alias_index.setdefault(alias, node)
    return alias_index


def _save_graph_snapshot(graph: Dict[str, Any]) -> Path:
    p = _snapshot_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": list(graph.get("nodes", [])),
        "edges": list(graph.get("edges", [])),
        "doc_index": dict(graph.get("doc_index", {})),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Graph snapshot saved: %s", p)
    return p


def _load_graph_snapshot() -> Optional[Dict[str, Any]]:
    p = _snapshot_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read graph snapshot %s: %s", p, exc)
        return None
    nodes = list(raw.get("nodes", []))
    edges = list(raw.get("edges", []))
    if not nodes:
        return None
    doc_index = dict(raw.get("doc_index", {}))
    # Rebuild parent/sibling metadata if missing
    parent_to_children: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        md = dict(n.get("metadata") or {})
        parent_id = md.get("parent_id")
        if parent_id:
            parent_to_children[str(parent_id)].append(str(n.get("node_id")))
        doc_id = str(md.get("doc_id") or "")
        if doc_id and doc_id not in doc_index:
            doc_index[doc_id] = {
                "doc_id": doc_id,
                "official_title": md.get("official_title") or md.get("law_name") or "",
                "doc_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "law_name": md.get("law_name") or md.get("official_title") or "",
                "law_type": md.get("law_type") or md.get("doc_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "year": md.get("year") or 0,
                "lead_block": md.get("lead_block") or "",
                "doc_number": md.get("doc_number"),
                "issuing_agency": md.get("issuing_agency"),
            }
    sibling_map: Dict[str, List[str]] = {}
    for _, children in parent_to_children.items():
        for child in children:
            sibling_map[child] = [x for x in children if x != child]
    for n in nodes:
        md = n.setdefault("metadata", {})
        node_id = str(n.get("node_id") or "")
        md["children_ids"] = parent_to_children.get(node_id, md.get("children_ids", []))
        md["sibling_ids"] = sibling_map.get(node_id, md.get("sibling_ids", []))
    node_index = {str(n["node_id"]): n for n in nodes if n.get("node_id")}
    alias_index = _build_alias_index(nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "node_index": node_index,
        "alias_index": alias_index,
        "doc_index": doc_index,
        "adjacency": _build_adjacency(edges),
        "reverse_adjacency": _build_reverse_adjacency(edges),
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


def _prefix_chunks(chunks: List[Dict[str, Any]], doc_prefix: str, *, doc_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        item = dict(ch)
        md = dict(item.get("metadata", {}) or {})

        for key in ["node_id", "chunk_id", "parent_id"]:
            val = md.get(key)
            if val:
                md[key] = f"{doc_prefix}::{val}"

        for list_key in ["children_ids", "sibling_ids"]:
            vals = md.get(list_key)
            if isinstance(vals, list):
                md[list_key] = [f"{doc_prefix}::{x}" for x in vals if x]

        md["doc_id"] = doc_prefix
        if doc_meta:
            for k, v in doc_meta.items():
                if md.get(k) in (None, "") and v not in (None, ""):
                    md[k] = v

        item["metadata"] = md
        out.append(item)
    return out


def _build_sibling_map(parent_to_children: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sibling_map: Dict[str, List[str]] = {}
    for _, children in parent_to_children.items():
        for child in children:
            sibling_map[child] = [x for x in children if x != child]
    return sibling_map


def build_runtime_graph(
    input_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    *,
    max_workers: int = 4,
) -> Dict[str, Any]:
    _ = max_workers
    start = time.perf_counter()
    _RUNTIME["graph_build_in_progress"] = True

    input_dir = input_dir or settings.normalized_dir
    glob_pattern = glob_pattern or settings.normalized_glob
    logger.info("Graph build started: input_dir=%s glob=%s", input_dir, glob_pattern)

    base = Path(input_dir)
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    try:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        doc_index: Dict[str, Dict[str, Any]] = {}
        parent_to_children: Dict[str, List[str]] = defaultdict(list)

        for fp in sorted(base.glob(glob_pattern)):
            if fp.suffix.lower() not in {".pdf", ".txt"}:
                continue

            text = load_document(fp)
            header = extract_document_header(text, fallback_name=fp.stem)
            doc_prefix = _slugify(fp.stem)
            doc_meta = header.to_metadata()
            doc_meta["doc_id"] = doc_prefix
            doc_meta["file_name"] = fp.name
            doc_meta["source_path"] = str(fp)
            doc_index[doc_prefix] = doc_meta

            chunks = legal_chunk(text, fallback_doc_name=fp.stem)
            chunks = _prefix_chunks(chunks, doc_prefix, doc_meta=doc_meta)

            for ch in chunks:
                md = dict(ch.get("metadata") or {})
                node_id = str(md.get("node_id") or md.get("chunk_id") or "")
                if not node_id:
                    continue
                node = {
                    "node_id": node_id,
                    "node_type": md.get("node_type") or "text",
                    "text": str(ch.get("text") or "").strip(),
                    "retrieval_text": str(ch.get("retrieval_text") or ch.get("text") or "").strip(),
                    "rerank_text": str(ch.get("rerank_text") or ch.get("retrieval_text") or ch.get("text") or "").strip(),
                    "metadata": md,
                }
                nodes.append(node)
                parent_id = md.get("parent_id")
                if parent_id:
                    edges.append({"source_id": str(parent_id), "target_id": node_id, "relation_type": "HAS_CHILD"})
                    parent_to_children[str(parent_id)].append(node_id)

        nodes = _dedup_nodes(nodes)
        edges = _dedup_edges(edges)
        sibling_map = _build_sibling_map(parent_to_children)
        node_index = {str(n["node_id"]): n for n in nodes if n.get("node_id")}
        for node_id, node in node_index.items():
            md = node.setdefault("metadata", {})
            md["children_ids"] = parent_to_children.get(node_id, md.get("children_ids", []))
            md["sibling_ids"] = sibling_map.get(node_id, md.get("sibling_ids", []))
            doc_id = str(md.get("doc_id") or "")
            if doc_id in doc_index:
                for key, val in doc_index[doc_id].items():
                    if md.get(key) in (None, "") and val not in (None, ""):
                        md[key] = val
        alias_index = _build_alias_index(nodes)
        graph = {
            "nodes": nodes,
            "edges": edges,
            "node_index": node_index,
            "alias_index": alias_index,
            "doc_index": doc_index,
            "adjacency": _build_adjacency(edges),
            "reverse_adjacency": _build_reverse_adjacency(edges),
        }
        _apply_runtime_graph(graph, built_seconds=time.perf_counter() - start)
        _save_graph_snapshot(graph)
        logger.info(
            "Graph build finished: docs=%s nodes=%s edges=%s aliases=%s took=%.2fs",
            len(doc_index), len(nodes), len(edges), len(alias_index), _RUNTIME["last_graph_build_seconds"]
        )
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
    graph = _RUNTIME["graph"]
    return {
        "graph_loaded": graph is not None,
        "graph_loaded_at": _RUNTIME["graph_loaded_at"],
        "graph_doc_count": _RUNTIME["graph_doc_count"],
        "graph_node_count": len((graph or {}).get("nodes", [])),
        "graph_edge_count": len((graph or {}).get("edges", [])),
        "graph_alias_count": len((graph or {}).get("alias_index", {})),
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
        "graph_aliases": len(graph.get("alias_index", {})),
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

        text = load_document(fp)
        header = extract_document_header(text, fallback_name=fp.stem)
        doc_title = fp.stem
        doc_prefix = _slugify(doc_title)
        doc_meta = header.to_metadata()
        doc_meta["doc_id"] = doc_prefix

        chunks = legal_chunk(text, max_chars=max_chars, overlap=overlap, fallback_doc_name=fp.stem)
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

        chunks = _prefix_chunks(chunks, doc_prefix, doc_meta=doc_meta)
        n = upsert_chunks(chunks, embeddings=embeddings, meta=doc_meta)
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
        "status": get_runtime_status(),
    }
