from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import Distance, VectorParams

from src.app.settings import settings

DEFAULT_VECTOR_SIZE = 1536


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def ensure_collection(
    client: QdrantClient,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> None:
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if settings.qdrant_collection in names:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def build_filter(filters: Dict[str, Any]) -> Optional[qm.Filter]:
    if not filters:
        return None

    must: List[qm.FieldCondition] = []

    for k, v in filters.items():
        if v is None or v == "":
            continue

        if isinstance(v, (int, float, bool, str)):
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        elif isinstance(v, list) and v:
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))

    if not must:
        return None

    return qm.Filter(must=must)


def upsert_points(
    client: QdrantClient,
    points: List[qm.PointStruct],
) -> None:
    if not points:
        return

    # Qdrant HTTP giới hạn payload request (mặc định thường ~32MB).
    # Chia batch để tránh lỗi 400 "JSON payload ... is larger than allowed".
    max_batch_bytes = 8 * 1024 * 1024  # 8MB an toàn hơn nhiều so với giới hạn 32MB
    max_batch_points = 256

    def _estimate_point_bytes(p: qm.PointStruct) -> int:
        payload = getattr(p, "payload", {}) or {}
        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception:
            payload_bytes = 0

        vec = getattr(p, "vector", None)
        if isinstance(vec, list):
            vector_bytes = len(vec) * 4  # float32 xấp xỉ
        else:
            vector_bytes = 0

        # overhead JSON + id + keys
        return payload_bytes + vector_bytes + 2048

    batch: List[qm.PointStruct] = []
    batch_bytes = 0

    for p in points:
        p_bytes = _estimate_point_bytes(p)

        if batch and (batch_bytes + p_bytes > max_batch_bytes or len(batch) >= max_batch_points):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=batch,
            )
            batch = []
            batch_bytes = 0

        batch.append(p)
        batch_bytes += p_bytes

    if batch:
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
        )


def search_qdrant(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> List[qm.ScoredPoint]:
    qfilter = build_filter(filters or {})
    return client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )
