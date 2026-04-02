from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from src.app.settings import settings

DEFAULT_VECTOR_SIZE = 1536


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )


def ensure_collection(client: QdrantClient, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
    collections = client.get_collections().collections
    names = {collection.name for collection in collections}
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
    for key, value in filters.items():
        if value in (None, ""):
            continue
        if isinstance(value, (int, float, bool, str)):
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        elif isinstance(value, list) and value:
            must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=value)))
    return qm.Filter(must=must) if must else None


def upsert_points(client: QdrantClient, points: List[qm.PointStruct]) -> None:
    if not points:
        return
    max_batch_bytes = 8 * 1024 * 1024
    max_batch_points = 256

    def _estimate_point_bytes(point: qm.PointStruct) -> int:
        payload = getattr(point, "payload", {}) or {}
        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception:
            payload_bytes = 0
        vector = getattr(point, "vector", None)
        vector_bytes = len(vector) * 4 if isinstance(vector, list) else 0
        return payload_bytes + vector_bytes + 2048

    batch: List[qm.PointStruct] = []
    batch_bytes = 0
    for point in points:
        point_bytes = _estimate_point_bytes(point)
        if batch and (batch_bytes + point_bytes > max_batch_bytes or len(batch) >= max_batch_points):
            client.upsert(collection_name=settings.qdrant_collection, points=batch)
            batch = []
            batch_bytes = 0
        batch.append(point)
        batch_bytes += point_bytes
    if batch:
        client.upsert(collection_name=settings.qdrant_collection, points=batch)


def search_qdrant(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> List[qm.ScoredPoint]:
    qfilter = build_filter(filters or {})
    if hasattr(client, "search"):
        return client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=False,
        )

    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", None)
        if points is not None:
            return list(points)
        if isinstance(response, dict):
            return list(response.get("points", []) or [])
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) != 404:
            raise

    legacy_response = client.http.search_api.search_points(
        collection_name=settings.qdrant_collection,
        search_request=qm.SearchRequest(
            vector=query_vector,
            limit=top_k,
            filter=qfilter,
            with_payload=True,
            with_vector=False,
        ),
    )
    result = getattr(legacy_response, "result", None)
    if result is not None:
        return list(result)
    if isinstance(legacy_response, dict):
        return list(legacy_response.get("result", []) or [])
    return []
