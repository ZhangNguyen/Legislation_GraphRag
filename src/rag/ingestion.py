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

    artifact = str(merged_meta.get("artifact_type") or "evidence")
    title = _norm_space(str(merged_meta.get("official_title") or merged_meta.get("law_name") or ""))
    path_title = _norm_space(str(merged_meta.get("path_title") or merged_meta.get("title") or ""))
    text = _norm_space(str(chunk.get("text") or ""))

    parts: List[str] = []
    if title:
        parts.append(f"Văn bản: {title}")
    if path_title:
        parts.append(f"Vị trí: {path_title}")
    if artifact == "doc_sketch":
        parts.append(f"Tóm tắt cấu trúc: {text}")
    elif artifact == "article_bundle":
        parts.append(f"Nội dung điều: {text}")
    else:
        parts.append(f"Nội dung: {text}")
    return "\n".join(p for p in parts if _norm_space(p)).strip()


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
        rerank_text = _norm_space(str(chunk.get("rerank_text") or retrieval_text))

        try:
            vector = embeddings.embed_query(retrieval_text)
        except Exception as exc:
            logger.warning("Embedding failed for chunk_id=%s: %s", chunk_id, exc)
            continue

        payload = {
            "text": text,
            "snippet": str(chunk.get("snippet") or text[:600]),
            "retrieval_text": retrieval_text,
            "rerank_text": rerank_text,
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
    for chunk in chunks:
        md = dict(chunk.get("metadata") or {})
        node_id = str(md.get("node_id") or md.get("chunk_id") or "")
        if not node_id:
            continue
        graph_nodes.append(
            {
                "node_id": node_id,
                "node_type": md.get("node_type") or "text",
                "text": str(chunk.get("text") or "").strip(),
                "retrieval_text": str(chunk.get("retrieval_text") or chunk.get("text") or "").strip(),
                "rerank_text": str(chunk.get("rerank_text") or chunk.get("retrieval_text") or chunk.get("text") or "").strip(),
                "metadata": md,
            }
        )
        parent_id = md.get("parent_id")
        if parent_id:
            graph_edges.append({"source_id": str(parent_id), "target_id": node_id, "relation_type": "HAS_CHILD"})

    return {
        "doc_meta": parsed.header.to_metadata(),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "chunks": chunks,
    }
