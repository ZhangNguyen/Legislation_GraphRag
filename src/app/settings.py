from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _split_csv(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "legal_chunks")

    normalized_dir: str = os.getenv("NORMALIZED_DIR", "data/normalized")
    normalized_glob: str = os.getenv("NORMALIZED_GLOB", "*.*")
    graph_snapshot_path: str = os.getenv("GRAPH_SNAPSHOT_PATH", "outputs/runtime/runtime_graph_snapshot.json")

    qdrant_top_k: int = int(os.getenv("QDRANT_TOP_K", "40"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "12"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "8"))
    cross_top_k: int = int(os.getenv("CROSS_TOP_K", "8"))
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.58"))

    bundle_max_children: int = int(os.getenv("BUNDLE_MAX_CHILDREN", "3"))
    bundle_max_siblings: int = int(os.getenv("BUNDLE_MAX_SIBLINGS", "2"))

    bm25_stats_path: str = os.getenv("BM25_STATS_PATH", "outputs/bm25/bm25_stats.json")

    answer_max_context_passages: int = int(os.getenv("ANSWER_MAX_CONTEXT_PASSAGES", "6"))
    answer_max_source_items: int = int(os.getenv("ANSWER_MAX_SOURCE_ITEMS", "6"))

    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "12"))

    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    cors_origins: List[str] = _split_csv(os.getenv("CORS_ORIGINS", "*"))

    poppler_path: str = os.getenv("POPPLER_PATH", "")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "vie")
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))


settings = Settings()
