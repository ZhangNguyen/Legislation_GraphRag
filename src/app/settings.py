from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _split_csv(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    hf_token: str = os.getenv("HF_TOKEN", "")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "legal_chunks")

    normalized_dir: str = os.getenv("NORMALIZED_DIR", "data/normalized")
    raw_dir: str = os.getenv("RAW_DIR", "data/raw")
    normalized_glob: str = os.getenv("NORMALIZED_GLOB", "*.*")
    graph_snapshot_path: str = os.getenv("GRAPH_SNAPSHOT_PATH", "outputs/runtime/runtime_graph_snapshot.json")
    bm25_stats_path: str = os.getenv("BM25_STATS_PATH", "outputs/bm25/bm25_stats.json")

    doc_top_k: int = _get_int("DOC_TOP_K", 3)
    content_top_k_per_doc: int = _get_int("CONTENT_TOP_K_PER_DOC", 3)
    final_top_k: int = _get_int("FINAL_TOP_K", 5)
    cross_top_k: int = _get_int("CROSS_TOP_K", 8)
    reference_hybrid_threshold: float = _get_float("REFERENCE_HYBRID_THRESHOLD", 0.85)
    doc_sketch_hybrid_threshold: float = _get_float("DOC_SKETCH_HYBRID_THRESHOLD", 0.82)
    cross_rerank_min_score: float = _get_float("CROSS_RERANK_MIN_SCORE", 0.05)
    hybrid_alpha: float = _get_float("HYBRID_ALPHA", 0.55)
    rrf_k: int = _get_int("RRF_K", 60)
    th2_passage_top_k: int = _get_int("TH2_PASSAGE_TOP_K", 12)

    answer_max_context_passages: int = _get_int("ANSWER_MAX_CONTEXT_PASSAGES", 5)
    answer_max_source_items: int = _get_int("ANSWER_MAX_SOURCE_ITEMS", 5)

    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")
    rerank_top_n: int = _get_int("RERANK_TOP_N", 12)

    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = _get_int("APP_PORT", 8000)
    cors_origins: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cors_origins", _split_csv(os.getenv("CORS_ORIGINS", "*")))


settings = Settings()
