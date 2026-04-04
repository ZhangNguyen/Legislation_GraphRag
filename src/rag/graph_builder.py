from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.app.settings import settings
from src.rag.chunking_legal import legal_chunk
from src.rag.ingestion import ingest_document, upsert_chunks
from src.rag.legal_versioning import annotate_graph_with_versioning, build_evidence_ref_index, build_version_map
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document

logger = logging.getLogger(__name__)

REFERENCE_NODE_TYPES_STRICT = {"article", "clause", "point"}
REFERENCE_NODE_TYPES_EXTENDED = REFERENCE_NODE_TYPES_STRICT | {"section", "item", "bullet"}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _unique_dicts(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        sig = tuple(str(item.get(k, "")).strip().lower() for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def _index_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(n["node_id"]): n for n in nodes if n.get("node_id")}


def _build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        if e.get("source_id"):
            adj[str(e["source_id"])].append(e)
    return dict(adj)


def _build_reverse_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rev: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        if e.get("target_id"):
            rev[str(e["target_id"])].append(e)
    return dict(rev)


def refresh_graph_indexes(graph: Dict[str, Any]) -> Dict[str, Any]:
    graph["node_index"] = _index_nodes(list(graph.get("nodes", [])))
    graph["adjacency"] = _build_adjacency(list(graph.get("edges", [])))
    graph["reverse_adjacency"] = _build_reverse_adjacency(list(graph.get("edges", [])))
    return graph


def build_graph(
    graph_nodes: List[Dict[str, Any]],
    entity_nodes: Optional[List[Dict[str, Any]]] = None,
    summary_nodes: Optional[List[Dict[str, Any]]] = None,
    graph_edges: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    entity_nodes = entity_nodes or []
    summary_nodes = summary_nodes or []
    graph_edges = graph_edges or []
    all_nodes = _unique_dicts(list(graph_nodes) + list(entity_nodes) + list(summary_nodes), ["node_id"])
    all_edges = _unique_dicts(
        list(graph_edges),
        ["source_id", "target_id", "relation_type", "source_text", "target_text", "target_article", "target_clause", "target_point"],
    )
    graph = {"nodes": all_nodes, "edges": all_edges}
    return refresh_graph_indexes(graph)


def materialize_graph_from_ingestion(ingestion_output: Dict[str, Any]) -> Dict[str, Any]:
    graph = build_graph(
        graph_nodes=ingestion_output.get("graph_nodes", []),
        entity_nodes=ingestion_output.get("entity_nodes", []),
        summary_nodes=ingestion_output.get("summary_nodes", []),
        graph_edges=ingestion_output.get("graph_edges", []),
    )
    return refresh_graph_indexes(graph)


def _snapshot_path() -> Path:
    p = Path(settings.graph_snapshot_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _collect_node_children(node: Dict[str, Any]) -> List[str]:
    md = dict(node.get("metadata") or {})
    return [str(x) for x in (md.get("children_ids") or []) if str(x).strip()]


def _node_type(node: Dict[str, Any]) -> str:
    md = dict(node.get("metadata") or {})
    return str(node.get("node_type") or md.get("node_type") or "").strip().lower()


def _node_md(node: Dict[str, Any]) -> Dict[str, Any]:
    return dict(node.get("metadata") or {})


def _is_reference_capable(node: Dict[str, Any]) -> bool:
    node_type = _node_type(node)
    children = _collect_node_children(node)
    md = _node_md(node)
    if node_type in REFERENCE_NODE_TYPES_STRICT:
        return True
    if node_type in REFERENCE_NODE_TYPES_EXTENDED and children:
        return True
    if children and (_norm_space(str(md.get("path_title") or "")) or _norm_space(str(node.get("text") or ""))):
        return True
    return False


def _is_section_anchor(node: Dict[str, Any]) -> bool:
    node_type = _node_type(node)
    return node_type in {"article", "clause", "point", "section", "item", "bullet"} and _is_reference_capable(node)


def _descendants(graph: Dict[str, Any], node_id: str) -> List[str]:
    node_index = graph.get("node_index", {}) or {}
    out: List[str] = []
    seen: Set[str] = set()
    queue = [node_id]
    while queue:
        cur = queue.pop(0)
        node = node_index.get(cur)
        if not node:
            continue
        children = _collect_node_children(node)
        for child_id in children:
            if child_id in seen:
                continue
            seen.add(child_id)
            out.append(child_id)
            queue.append(child_id)
    return out


def collect_subtree_nodes(graph: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    node_index = graph.get("node_index", {}) or {}
    ids = [node_id] + _descendants(graph, node_id)
    nodes = [node_index[nid] for nid in ids if nid in node_index]
    nodes.sort(key=lambda n: int((_node_md(n).get("order_index") or 0)))
    return nodes


def collect_subtree_text(graph: Dict[str, Any], node_id: str) -> str:
    parts: List[str] = []
    for node in collect_subtree_nodes(graph, node_id):
        text = _norm_space(str(node.get("text") or ""))
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def collect_amendment_source_nodes(graph: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    reverse_adj = graph.get("reverse_adjacency", {}) or {}
    node_index = graph.get("node_index", {}) or {}
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for edge in reverse_adj.get(node_id, []):
        rel = str(edge.get("relation_type") or "").upper()
        if rel not in {"AMENDS", "REPEALS", "REPLACES", "PARTIALLY_AMENDS", "PARTIALLY_REPEALS"}:
            continue
        src = str(edge.get("source_id") or "")
        if not src or src in seen or src not in node_index:
            continue
        seen.add(src)
        out.append(node_index[src])
    return out


def _build_doc_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    node_index = graph.get("node_index", {}) or {}
    docs: Dict[str, Dict[str, Any]] = {}
    for node_id, node in node_index.items():
        md = _node_md(node)
        doc_id = str(md.get("doc_id") or "").strip()
        if not doc_id:
            continue
        doc = docs.setdefault(
            doc_id,
            {
                "id": doc_id,
                "doc_id": doc_id,
                "official_title": str(md.get("official_title") or md.get("law_name") or "").strip(),
                "doc_sketch": "",
                "law_name": str(md.get("law_name") or md.get("official_title") or "").strip(),
                "law_type": str(md.get("law_type") or md.get("doc_type") or "Unknown"),
                "doc_type": str(md.get("doc_type") or md.get("law_type") or "Unknown"),
                "source": str(md.get("source") or md.get("issuing_agency") or "LocalFile"),
                "issuing_agency": str(md.get("issuing_agency") or md.get("source") or "").strip(),
                "year": int(md.get("year") or 0),
                "date_raw": str(md.get("date_raw") or "").strip(),
                "doc_number": str(md.get("doc_number") or "").strip(),
                "reference_node_ids_strict": [],
                "reference_node_ids": [],
                "evidence_node_ids": [],
                "member_node_ids": [],
                "doc_sketch_node_id": "",
            },
        )
        artifact = str(md.get("artifact_type") or "evidence")
        if artifact == "doc_sketch":
            doc["doc_sketch"] = str(node.get("text") or md.get("retrieval_text") or "")
            doc["doc_sketch_node_id"] = node_id
        elif artifact in {"evidence", "article_bundle"}:
            doc["evidence_node_ids"].append(node_id)
            if _is_reference_capable(node):
                doc["reference_node_ids"].append(node_id)
            if _node_type(node) in REFERENCE_NODE_TYPES_STRICT:
                doc["reference_node_ids_strict"].append(node_id)
        doc["member_node_ids"].append(node_id)

    for doc in docs.values():
        for key in ["reference_node_ids_strict", "reference_node_ids", "evidence_node_ids", "member_node_ids"]:
            seen: Set[str] = set()
            cleaned: List[str] = []
            for nid in doc[key]:
                if nid in seen:
                    continue
                seen.add(nid)
                cleaned.append(nid)
            doc[key] = cleaned
    return docs


def finalize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    graph = refresh_graph_indexes(graph)
    graph = annotate_graph_with_versioning(graph)
    graph = refresh_graph_indexes(graph)
    graph["doc_index"] = _build_doc_index(graph)
    graph["evidence_ref_index"] = build_evidence_ref_index(graph)
    graph["version_map"] = build_version_map(graph)
    return graph


def build_runtime_graph(*, input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    base = Path(input_dir or settings.normalized_dir)
    pattern = glob_pattern or settings.normalized_glob
    if not base.exists():
        raise RuntimeError(f"Missing normalized dir: {base}")

    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []
    for fp in sorted(base.glob(pattern)):
        if fp.suffix.lower() not in {".txt", ".pdf", ".doc", ".docx"}:
            continue
        text = load_document(fp)
        out = ingest_document(text, fallback_doc_name=fp.stem)
        all_nodes.extend(out.get("graph_nodes", []))
        all_edges.extend(out.get("graph_edges", []))

    graph = build_graph(graph_nodes=all_nodes, graph_edges=all_edges)
    return finalize_graph(graph)


def save_graph_snapshot(graph: Dict[str, Any]) -> Path:
    p = _snapshot_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": list(graph.get("nodes", [])),
        "edges": list(graph.get("edges", [])),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load_graph_snapshot() -> Optional[Dict[str, Any]]:
    p = _snapshot_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load graph snapshot: %s", exc)
        return None
    graph = build_graph(graph_nodes=list(raw.get("nodes", [])), graph_edges=list(raw.get("edges", [])))
    return finalize_graph(graph)


def reindex_qdrant_from_normalized(*, input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    base = Path(input_dir or settings.normalized_dir)
    pattern = glob_pattern or settings.normalized_glob
    if not base.exists():
        raise RuntimeError(f"Missing normalized dir: {base}")
    client = get_qdrant_client()
    ensure_collection(client)
    total = 0
    for fp in sorted(base.glob(pattern)):
        if fp.suffix.lower() not in {".txt", ".pdf", ".doc", ".docx"}:
            continue
        text = load_document(fp)
        chunks = legal_chunk(text, fallback_doc_name=fp.stem)
        total += upsert_chunks(chunks)
    return {"status": "ok", "indexed_chunks": total}


def rebuild_everything(*, input_dir: Optional[str] = None, glob_pattern: Optional[str] = None) -> Dict[str, Any]:
    reindex_result = reindex_qdrant_from_normalized(input_dir=input_dir, glob_pattern=glob_pattern)
    graph = build_runtime_graph(input_dir=input_dir, glob_pattern=glob_pattern)
    snapshot = save_graph_snapshot(graph)
    return {
        "status": "ok",
        "indexed_chunks": reindex_result.get("indexed_chunks", 0),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "snapshot_path": str(snapshot),
    }
