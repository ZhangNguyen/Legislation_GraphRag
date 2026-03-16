from __future__ import annotations

import uuid
from typing import Any, Dict, List

from qdrant_client.http import models as qm

from src.app.settings import settings
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.rag.chunking_legal import LegalChunk

def build_points(chunks: List[LegalChunk], vectors: List[List[float]], meta: Dict[str, Any]) -> List[qm.PointStruct]:
    points: List[qm.PointStruct] = []
    for idx, (ch,vec) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{meta['law_name']}_{meta['year']}_{(ch.article or 'na')}_{(ch.clause or 'na')}_{idx}"
        payload = {
            "text": ch.text,
            "snippet": (ch.text[:300] + "...") if len(ch.text) > 300 else ch.text,
            "chunk_id": chunk_id,
            "law_name": meta["law_name"],
            "law_type": meta["law_type"],
            "year": int(meta["year"]),
            "source": meta["source"],
            "article": ch.article,
            "clause": ch.clause,
            "point": ch.point,
        }
        points.append(qm.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
    return points

def upsert_chunks(chunks: List[LegalChunk], embeddings, meta: Dict[str, Any]) -> int:
    client = get_qdrant_client()
    ensure_collection(client)

    texts = [c.text for c in chunks]
    vectors = embeddings.embed_documents(texts)
    points = build_points(chunks, vectors, meta)

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)

