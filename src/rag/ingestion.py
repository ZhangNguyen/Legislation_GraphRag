from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qm

from src.rag.chunking_legal import legal_chunk, parse_legal_document
from src.rag.openai_clients import get_embedings
from src.storage.qdrant_store import get_qdrant_client, upsert_points

logger = logging.getLogger(__name__)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def _build_retrieval_text(chunk: Dict[str, Any], merged_meta: Dict[str, Any]) -> str:
    explicit = _norm_space(str(chunk.get("retrieval_text") or ""))
    if explicit:
        return explicit

    parts: List[str] = []
    for key in [
        "doc_type",
        "official_title",
        "issuing_agency",
        "doc_number",
        "lead_block",
        "path_title",
        "node_type",
        "article",
        "clause",
        "point",
    ]:
        value = merged_meta.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    parts.append(str(chunk.get("text") or ""))
    return "\n".join(_norm_space(x) for x in parts if _norm_space(str(x))).strip()


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    *,
    embeddings=None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    if not chunks:
        return 0

    embeddings = embeddings or get_embedings()
    base_meta = dict(meta or {})
    points: List[qm.PointStruct] = []

    for idx, chunk in enumerate(chunks):
        text = _norm_space(str(chunk.get("text") or ""))
        if not text:
            continue

        chunk_meta = dict(chunk.get("metadata") or {})
        merged_meta = {**base_meta, **chunk_meta}
        chunk_id = str(merged_meta.get("chunk_id") or merged_meta.get("node_id") or f"chunk_{idx}")
        merged_meta["chunk_id"] = chunk_id
        merged_meta["node_id"] = str(merged_meta.get("node_id") or chunk_id)
        merged_meta["year"] = _safe_int(merged_meta.get("year"), 0)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

        retrieval_text = _build_retrieval_text(chunk, merged_meta)
        try:
            vector = embeddings.embed_query(retrieval_text)
        except Exception as exc:
            logger.warning("Embedding failed for chunk_id=%s: %s", chunk_id, exc)
            continue

        payload = {
            "text": text,
            "snippet": str(chunk.get("snippet") or text[:600]),
            "retrieval_text": retrieval_text,
            "rerank_text": str(chunk.get("rerank_text") or retrieval_text),
            "metadata": merged_meta,
            **merged_meta,
        }
        points.append(qm.PointStruct(id=point_id, vector=vector, payload=payload))

    if not points:
        return 0

    client = get_qdrant_client()
    upsert_points(client, points)
    return len(points)


def ingest_document(text: str, *, fallback_doc_name: str = "") -> Dict[str, Any]:
    parsed = parse_legal_document(text, fallback_doc_name=fallback_doc_name)
    chunks = legal_chunk(text, fallback_doc_name=fallback_doc_name)

    graph_nodes: List[Dict[str, Any]] = []
    graph_edges: List[Dict[str, Any]] = []
    for node in parsed.nodes:
        metadata = {
            "doc_id": parsed.header.doc_id,
            "law_name": parsed.header.official_title or parsed.header.file_stem,
            "law_type": parsed.header.doc_type or "Unknown",
            "source": parsed.header.issuing_agency or "LocalFile",
            "year": parsed.header.year or 0,
            "chunk_id": node.node_id,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "path_title": node.path_title,
            "article": node.article,
            "clause": node.clause,
            "point": node.point,
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids),
            "official_title": parsed.header.official_title or parsed.header.file_stem,
            "doc_type": parsed.header.doc_type or "Unknown",
            "lead_block": parsed.header.lead_block,
        }
        graph_nodes.append(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "text": node.text,
                "metadata": metadata,
            }
        )
        if node.parent_id:
            graph_edges.append(
                {
                    "source_id": node.parent_id,
                    "target_id": node.node_id,
                    "relation_type": "HAS_CHILD",
                }
            )

    return {
        "doc_meta": parsed.header.to_metadata(),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "chunks": chunks,
    }
