from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from src.app.settings import settings
from src.rag.chunking_legal import legal_chunk
from src.rag.document_header import extract_document_header
from src.rag.ingestion import upsert_chunks
from src.rag.openai_clients import get_embedings, get_llm
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document

try:
    from src.rag.chunking_legal import semantic_merge_safe
except Exception:
    semantic_merge_safe = None

try:
    from src.rag.rerank_cross import get_cross_encoder
except Exception:
    get_cross_encoder = None


_RUNTIME: Dict[str, Any] = {
    "graph": None,
    "graph_loaded_at": None,
    "graph_doc_count": 0,
    "last_graph_build_seconds": None,
    "last_reindex_seconds": None,
    "graph_build_in_progress": False,
    "services_warmed_up": False,
    "services_warmup_in_progress": False,
    "last_services_warmup_seconds": None,
    "startup_preload_completed": False,
    "startup_preload_at": None,
    "last_startup_preload_seconds": None,
    "warmup_last_error": None,
}

logger = logging.getLogger(__name__)
_RUNTIME_LOCK = Lock()

SUMMARY_EDGE = "HAS_CHILD_SUMMARY"
SUMMARIZES_EDGE = "SUMMARIZES"


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


def _rebuild_runtime_maps(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    graph["node_index"] = {str(n["node_id"]): n for n in nodes if n.get("node_id")}
    graph["alias_index"] = _build_alias_index(nodes)
    graph["adjacency"] = _build_adjacency(edges)
    graph["reverse_adjacency"] = _build_reverse_adjacency(edges)
    return graph


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

    graph = {
        "nodes": nodes,
        "edges": edges,
        "doc_index": doc_index,
    }
    graph = _augment_graph_with_runtime_summaries(graph)
    graph = _rebuild_runtime_maps(graph)
    return graph


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

        for list_key in ["children_ids", "sibling_ids", "source_node_ids"]:
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


def _graph_has_summary_nodes(graph: Dict[str, Any], *, doc_id: Optional[str] = None) -> bool:
    for node in graph.get("nodes", []) or []:
        md = node.get("metadata", {}) or {}
        if md.get("artifact_type") != "summary":
            continue
        if doc_id is None or str(md.get("doc_id") or "") == str(doc_id):
            return True
    return False


def _candidate_doc_summary(doc_id: str, graph: Dict[str, Any], doc_meta: Dict[str, Any]) -> str:
    for node in graph.get("nodes", []) or []:
        md = node.get("metadata", {}) or {}
        if str(md.get("doc_id") or "") != doc_id:
            continue
        if str(md.get("artifact_type") or "").strip().lower() == "doc_sketch":
            text = _norm_space(str(node.get("text") or md.get("doc_summary") or ""))
            if text:
                return text
        text = _norm_space(str(md.get("doc_summary") or ""))
        if text:
            return text
    parts = [
        f"Loại văn bản: {str(doc_meta.get('doc_type') or doc_meta.get('law_type') or '').strip()}" if str(doc_meta.get('doc_type') or doc_meta.get('law_type') or '').strip() else '',
        f"Tiêu đề: {str(doc_meta.get('official_title') or doc_meta.get('law_name') or '').strip()}" if str(doc_meta.get('official_title') or doc_meta.get('law_name') or '').strip() else '',
        f"Số văn bản: {str(doc_meta.get('doc_number') or '').strip()}" if str(doc_meta.get('doc_number') or '').strip() else '',
    ]
    return _norm_space("\n".join(part for part in parts if part))


def _summary_text_from_evidence(node: Dict[str, Any]) -> str:
    return _norm_space(
        str(node.get("text") or node.get("retrieval_text") or node.get("rerank_text") or "")
    )


def _augment_graph_with_runtime_summaries(graph: Dict[str, Any]) -> Dict[str, Any]:
    existing_nodes = list(graph.get("nodes", []))
    existing_edges = list(graph.get("edges", []))
    doc_index = dict(graph.get("doc_index", {}))

    evidence_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in existing_nodes:
        md = node.get("metadata", {}) or {}
        if md.get("artifact_type") == "summary":
            continue
        if str(md.get("artifact_type") or "evidence").strip().lower() != "evidence":
            continue
        doc_id = str(md.get("doc_id") or "").strip()
        if doc_id:
            evidence_by_doc[doc_id].append(node)

    new_nodes: List[Dict[str, Any]] = []
    new_edges: List[Dict[str, Any]] = []

    for doc_id, evidence_nodes in evidence_by_doc.items():
        if _graph_has_summary_nodes(graph, doc_id=doc_id):
            continue

        doc_meta = dict(doc_index.get(doc_id) or {})
        evidence_ids = {str(n.get("node_id")) for n in evidence_nodes if n.get("node_id")}
        summary_id_map: Dict[str, str] = {}

        for node in evidence_nodes:
            node_id = str(node.get("node_id") or "").strip()
            if not node_id:
                continue
            summary_id_map[node_id] = f"{node_id}::summary"

        root_summary_ids: List[str] = []
        for node in evidence_nodes:
            node_id = str(node.get("node_id") or "").strip()
            if not node_id:
                continue
            md = dict(node.get("metadata") or {})
            summary_id = summary_id_map[node_id]
            parent_id = str(md.get("parent_id") or "").strip()
            child_ids = [
                summary_id_map[str(cid)]
                for cid in (md.get("children_ids") or [])
                if str(cid) in summary_id_map
            ]
            if not parent_id or parent_id not in evidence_ids:
                root_summary_ids.append(summary_id)

            summary_md = {
                **md,
                "doc_id": doc_id,
                "artifact_type": "summary",
                "summary_level": str(node.get("node_type") or md.get("node_type") or "text").strip().lower() or "text",
                "source_node_ids": [node_id],
                "child_summary_ids": child_ids,
                "title": md.get("path_title") or md.get("title") or md.get("article") or md.get("clause") or md.get("point") or md.get("section") or md.get("subsection") or node_id,
                "law_name": md.get("law_name") or md.get("official_title") or doc_meta.get("official_title") or doc_meta.get("law_name") or "",
                "law_type": md.get("law_type") or md.get("doc_type") or doc_meta.get("doc_type") or doc_meta.get("law_type") or "Unknown",
            }
            new_nodes.append(
                {
                    "node_id": summary_id,
                    "node_type": "summary",
                    "text": _summary_text_from_evidence(node),
                    "retrieval_text": _summary_text_from_evidence(node),
                    "rerank_text": _summary_text_from_evidence(node),
                    "metadata": summary_md,
                }
            )
            new_edges.append(
                {
                    "source_id": summary_id,
                    "target_id": node_id,
                    "relation_type": SUMMARIZES_EDGE,
                }
            )
            if parent_id and parent_id in summary_id_map:
                new_edges.append(
                    {
                        "source_id": summary_id_map[parent_id],
                        "target_id": summary_id,
                        "relation_type": SUMMARY_EDGE,
                    }
                )

        doc_summary_text = _candidate_doc_summary(doc_id, graph, doc_meta)
        if doc_summary_text:
            doc_summary_id = f"{doc_id}::summary::document"
            doc_title = _norm_space(
                str(doc_meta.get("official_title") or doc_meta.get("law_name") or doc_meta.get("file_name") or doc_id)
            )
            doc_summary_md = {
                **doc_meta,
                "doc_id": doc_id,
                "artifact_type": "summary",
                "summary_level": "document",
                "title": doc_title or doc_id,
                "document_title": doc_title or doc_id,
                "law_name": doc_meta.get("official_title") or doc_meta.get("law_name") or doc_title or doc_id,
                "law_type": doc_meta.get("doc_type") or doc_meta.get("law_type") or "Unknown",
                "child_summary_ids": sorted(set(root_summary_ids)),
                "source_node_ids": [],
            }
            new_nodes.append(
                {
                    "node_id": doc_summary_id,
                    "node_type": "summary",
                    "text": doc_summary_text,
                    "retrieval_text": doc_summary_text,
                    "rerank_text": doc_summary_text,
                    "metadata": doc_summary_md,
                }
            )
            for child_summary_id in sorted(set(root_summary_ids)):
                new_edges.append(
                    {
                        "source_id": doc_summary_id,
                        "target_id": child_summary_id,
                        "relation_type": SUMMARY_EDGE,
                    }
                )
            doc_index.setdefault(doc_id, {})["doc_summary_node_id"] = doc_summary_id

    if not new_nodes and not new_edges:
        return _rebuild_runtime_maps(graph)

    graph["nodes"] = _dedup_nodes(existing_nodes + new_nodes)
    graph["edges"] = _dedup_edges(existing_edges + new_edges)
    graph["doc_index"] = doc_index
    return _rebuild_runtime_maps(graph)


def warmup_runtime_services(force: bool = False) -> Dict[str, Any]:
    if _RUNTIME.get("services_warmed_up") and not force:
        return {
            "services_warmed_up": True,
            "seconds": _RUNTIME.get("last_services_warmup_seconds"),
            "error": _RUNTIME.get("warmup_last_error"),
        }

    with _RUNTIME_LOCK:
        if _RUNTIME.get("services_warmed_up") and not force:
            return {
                "services_warmed_up": True,
                "seconds": _RUNTIME.get("last_services_warmup_seconds"),
                "error": _RUNTIME.get("warmup_last_error"),
            }

        start = time.perf_counter()
        _RUNTIME["services_warmup_in_progress"] = True
        error_messages: List[str] = []
        warmed = {"llm": False, "embeddings": False, "cross_encoder": False}

        try:
            get_llm()
            warmed["llm"] = True
        except Exception as exc:
            logger.warning("LLM warmup failed: %s", exc)
            error_messages.append(f"llm: {exc}")

        try:
            get_embedings()
            warmed["embeddings"] = True
        except Exception as exc:
            logger.warning("Embeddings warmup failed: %s", exc)
            error_messages.append(f"embeddings: {exc}")

        if get_cross_encoder is not None:
            try:
                get_cross_encoder()
                warmed["cross_encoder"] = True
            except Exception as exc:
                logger.warning("Cross-encoder warmup failed: %s", exc)
                error_messages.append(f"cross_encoder: {exc}")

        seconds = time.perf_counter() - start
        _RUNTIME["last_services_warmup_seconds"] = seconds
        _RUNTIME["services_warmed_up"] = any(warmed.values()) and not error_messages
        _RUNTIME["warmup_last_error"] = "; ".join(error_messages) if error_messages else None
        _RUNTIME["services_warmup_in_progress"] = False
        return {
            "services_warmed_up": _RUNTIME["services_warmed_up"],
            "seconds": seconds,
            "details": warmed,
            "error": _RUNTIME["warmup_last_error"],
        }


def preload_runtime_on_startup(
    *,
    preload_graph_snapshot: bool = True,
    build_graph_if_missing: bool = False,
    warm_services: bool = True,
) -> Dict[str, Any]:
    start = time.perf_counter()
    graph_loaded = False
    built_graph = False
    warmup_info: Optional[Dict[str, Any]] = None

    if warm_services:
        warmup_info = warmup_runtime_services()

    if preload_graph_snapshot:
        graph_loaded = preload_runtime_graph_from_snapshot()

    if not graph_loaded and build_graph_if_missing:
        build_runtime_graph()
        graph_loaded = True
        built_graph = True

    seconds = time.perf_counter() - start
    _RUNTIME["startup_preload_completed"] = True
    _RUNTIME["startup_preload_at"] = time.time()
    _RUNTIME["last_startup_preload_seconds"] = seconds
    return {
        "graph_loaded": graph_loaded,
        "graph_built": built_graph,
        "warmup": warmup_info,
        "seconds": seconds,
    }


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
        graph = {
            "nodes": nodes,
            "edges": edges,
            "doc_index": doc_index,
        }
        graph = _augment_graph_with_runtime_summaries(graph)
        graph = _rebuild_runtime_maps(graph)
        _apply_runtime_graph(graph, built_seconds=time.perf_counter() - start)
        _save_graph_snapshot(graph)
        logger.info(
            "Graph build finished: docs=%s nodes=%s edges=%s aliases=%s took=%.2fs",
            len(doc_index), len(graph.get("nodes", [])), len(graph.get("edges", [])), len(graph.get("alias_index", {})), _RUNTIME["last_graph_build_seconds"]
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
    summary_count = 0
    if graph is not None:
        for node in graph.get("nodes", []) or []:
            md = node.get("metadata", {}) or {}
            if md.get("artifact_type") == "summary":
                summary_count += 1
    return {
        "graph_loaded": graph is not None,
        "graph_loaded_at": _RUNTIME["graph_loaded_at"],
        "graph_doc_count": _RUNTIME["graph_doc_count"],
        "graph_node_count": len((graph or {}).get("nodes", [])),
        "graph_edge_count": len((graph or {}).get("edges", [])),
        "graph_alias_count": len((graph or {}).get("alias_index", {})),
        "graph_summary_count": summary_count,
        "last_graph_build_seconds": _RUNTIME["last_graph_build_seconds"],
        "last_reindex_seconds": _RUNTIME["last_reindex_seconds"],
        "graph_build_in_progress": _RUNTIME["graph_build_in_progress"],
        "graph_snapshot_path": str(_snapshot_path()),
        "graph_snapshot_exists": _snapshot_path().exists(),
        "services_warmed_up": _RUNTIME["services_warmed_up"],
        "services_warmup_in_progress": _RUNTIME["services_warmup_in_progress"],
        "last_services_warmup_seconds": _RUNTIME["last_services_warmup_seconds"],
        "startup_preload_completed": _RUNTIME["startup_preload_completed"],
        "startup_preload_at": _RUNTIME["startup_preload_at"],
        "last_startup_preload_seconds": _RUNTIME["last_startup_preload_seconds"],
        "warmup_last_error": _RUNTIME["warmup_last_error"],
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
        "graph_summaries": sum(1 for n in graph.get("nodes", []) if (n.get("metadata", {}) or {}).get("artifact_type") == "summary"),
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
