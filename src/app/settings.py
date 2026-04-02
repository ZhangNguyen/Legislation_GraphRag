from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _split_csv(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


class Settings:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "legal_chunks")

        self.normalized_dir = os.getenv("NORMALIZED_DIR", "data/normalized")
        self.normalized_glob = os.getenv("NORMALIZED_GLOB", "*.*")
        self.graph_snapshot_path = os.getenv("GRAPH_SNAPSHOT_PATH", "outputs/runtime/runtime_graph_snapshot.json")

        self.qdrant_top_k = int(os.getenv("QDRANT_TOP_K", "20"))
        self.bm25_top_k = int(os.getenv("BM25_TOP_K", "12"))
        self.final_top_k = int(os.getenv("FINAL_TOP_K", "15"))
        self.cross_top_k = int(os.getenv("CROSS_TOP_K", "8"))

        self.bm25_stats_path = os.getenv("BM25_STATS_PATH", "outputs/bm25/bm25_stats.json")

        self.answer_max_context_passages = int(os.getenv("ANSWER_MAX_CONTEXT_PASSAGES", "50"))
        self.answer_max_source_items = int(os.getenv("ANSWER_MAX_SOURCE_ITEMS", "50"))

        self.rerank_model = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.rerank_device = os.getenv("RERANK_DEVICE", "cpu")
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "12"))

        self.app_host = os.getenv("APP_HOST", "127.0.0.1")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        self.cors_origins = _split_csv(os.getenv("CORS_ORIGINS", "*"))


settings = Settings()
