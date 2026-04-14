from __future__ import annotations

import os
import json
from typing import Any, Dict

from src.app.settings import settings
from src.rag.bm25_corpus import build_bm25_corpus


def load_or_build_bm25(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Load BM25 stats nếu đã tồn tại,
    nếu không thì build lại.
    """

    path = settings.bm25_stats_path

    # Nếu có file và không force → load
    if os.path.exists(path) and not force_rebuild:
        with open(path, "r", encoding="utf-8") as f:
            print("📥 Loading BM25 stats...")
            return json.load(f)

    # Nếu chưa có hoặc force → build
    print("🔨 Building BM25 corpus...")

    bm25_data = build_bm25_corpus(
        input_dir=settings.normalized_dir,
        glob_pattern=settings.normalized_glob,
    )

    # Lưu lại
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)

    print("💾 BM25 stats saved to:", path)

    return bm25_data