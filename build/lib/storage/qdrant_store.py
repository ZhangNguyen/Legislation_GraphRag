from qdrant_client import QdrantClient
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
