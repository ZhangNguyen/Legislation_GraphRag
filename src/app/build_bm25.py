from __future__ import annotations

import argparse

from src.app.settings import settings
from src.rag.bm25_corpus import build_bm25_stats_from_normalized, save_bm25_stats


def main() -> None:
    parser = argparse.ArgumentParser("Build BM25 stats from data/normalized")
    parser.add_argument("--input_dir", type=str, default=settings.normalized_dir)
    parser.add_argument("--glob", type=str, default=settings.normalized_glob)
    parser.add_argument("--output", type=str, default=settings.bm25_stats_path)
    parser.add_argument("--max_chars", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    stats = build_bm25_stats_from_normalized(
        input_dir=args.input_dir,
        glob_pattern=args.glob,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    save_bm25_stats(stats, args.output)
    print(f"[OK] Saved BM25 stats -> {args.output}")
    print(f"N={stats['N']}, avgdl={stats['avgdl']:.2f}, vocab={len(stats['idf'])}")


if __name__ == "__main__":
    main()
