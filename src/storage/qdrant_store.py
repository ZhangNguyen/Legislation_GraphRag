from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import Distance, VectorParams

from src.app.settings import settings

DEFAULT_VECTOR_SIZE = 1536

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,)

def ensure_collection(client: QdrantClient,vector_size: int = DEFAULT_VECTOR_SIZE)-> None:
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if settings.qdrant_collection in names:
        return
    print("Create new collection")
    client.create_collection(collection_name=settings.qdrant_collection,vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

def build_filter(filters:Dict[str,Any]) -> Optional[qm.Filter]:
    """
        filters cho payload Qdrant, ví dụ:
        {
          "law_name": "...",
          "year": 2015,
          "law_type": "...",
          "article": "Điều 5",
          "clause": "Khoản 1"
        }
        """
    if not filters:
        return None

    must: List[qm.FieldCondition] = []

    for k,v in filters.items():
        if v is None or v == "":
            continue
        if isinstance(v, (int, float,bool,str)):
            must.append(qm.FieldCondition(key=k,match=qm.MatchValue(value=v)))
        elif isinstance(v, list):
            must.append(qm.FieldCondition(key=k,match=qm.MatchValue(any=v)))
    if not must:
        return None
    return qm.Filter(must=must)

def search_qdrant(client:QdrantClient, query_vector: List[float], top_k:int, filters:Optional[Dict[str,Any]]=None) -> List[qm.ScoredPoint]:
    qfilter = build_filter(filters or {})
    return client.search(collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=False,)
