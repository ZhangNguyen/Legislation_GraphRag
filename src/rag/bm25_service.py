from __future__ import annotations

import os
import json
from typing import Any, Dict

from src.app.settings import settings
from src.rag.bm25_corpus import build_bm25_stats_from_normalized


def load_or_build_bm25(force_rebuild: bool = False) -> Dict[str, Any]:
    path = settings.bm25_stats_path

    if os.path.exists(path) and not force_rebuild:
        with open(path, "r", encoding="utf-8") as f:
            print("📥 Loading BM25 stats...")
            return json.load(f)

    print("🔨 Building BM25 stats...")

    bm25_data = build_bm25_stats_from_normalized(
        input_dir=settings.normalized_dir,
        glob_pattern=settings.normalized_glob,
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)

    print("💾 BM25 stats saved to:", path)
    return bm25_data