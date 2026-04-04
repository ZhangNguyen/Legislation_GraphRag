from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.app.settings import settings
from src.rag.chunking_legal import legal_chunk
from src.rag.ingestion import upsert_chunks
from src.rag.openai_clients import get_embedings
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document
from src.utils.normalize_raw import normalize_raw_to_normalized

try:
    from src.rag.chunking_legal import semantic_merge_safe
except Exception:
    semantic_merge_safe = None


def _ingest_single_file(fp: Path, args: argparse.Namespace) -> int:
    text = load_document(fp)
    print(f"[DEBUG] {fp.name}: text_len={len(text)}")

    chunks = legal_chunk(
        text,
        max_chars=args.max_chars,
        overlap=args.overlap,
        fallback_doc_name=fp.stem,
    )
    print(f"[DEBUG] {fp.name}: chunks={len(chunks)}")

    if not chunks:
        print(f"[WARN] {fp.name}: no chunks -> SKIP")
        return 0

    embeddings = get_embedings()

    if args.semantic_merge:
        if semantic_merge_safe is None:
            raise RuntimeError("semantic_merge_safe not available in src.rag.chunking_legal")
        chunks = semantic_merge_safe(
            chunks,
            embeddings=embeddings,
            min_chars=args.min_chars,
            sim_threshold=args.sim_threshold,
            max_merged_chars=args.max_merged_chars,
        )
        print(f"[DEBUG] {fp.name}: chunks_after_semantic_merge={len(chunks)}")

    # Không prefix lại node_id/chunk_id ở đây.
    # legal_chunk()/parse_legal_document() đã sinh id theo doc_id chuẩn.
    # Không truyền meta cứng để tránh đè metadata parser.
    n = upsert_chunks(
        chunks,
        embeddings=embeddings,
        meta={},
    )
    print(f"[OK] {fp.name}: upserted {n} chunks")
    return int(n)


def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into Qdrant (NO OCR).")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data/raw -> data/normalized trước khi ingest.",
    )
    parser.add_argument("--input_dir", type=str, default=settings.normalized_dir)
    parser.add_argument("--glob", type=str, default=settings.normalized_glob)
    parser.add_argument("--max_chars", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=120)

    parser.add_argument("--semantic_merge", action="store_true")
    parser.add_argument("--min_chars", type=int, default=350)
    parser.add_argument("--sim_threshold", type=float, default=0.88)
    parser.add_argument("--max_merged_chars", type=int, default=1600)
    parser.add_argument(
        "--upsert_workers",
        type=int,
        default=1,
        help="Số luồng upsert song song theo file (vd: 4).",
    )

    args = parser.parse_args()

    if args.normalize:
        normalize_raw_to_normalized()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Missing folder: {input_dir}")

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No files matched: {input_dir}/{args.glob}")

    files = [fp for fp in files if fp.suffix.lower() in {".pdf", ".txt"}]
    if not files:
        raise RuntimeError(f"No supported files (.pdf/.txt) matched: {input_dir}/{args.glob}")

    client = get_qdrant_client()
    ensure_collection(client)

    total = 0
    workers = max(1, int(args.upsert_workers))

    if workers == 1:
        for fp in files:
            total += _ingest_single_file(fp, args)
    else:
        print(f"[INFO] Running upsert with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_ingest_single_file, fp, args): fp for fp in files}
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    total += int(future.result())
                except Exception as exc:
                    print(f"[ERROR] {fp.name}: {exc}")

    print("DONE. Total chunks:", total)


if __name__ == "__main__":
    main()