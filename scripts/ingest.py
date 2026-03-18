# import argparse
# from pathlib import Path
#
# from src.rag.openai_clients import get_embedings
# from src.utils.loader import load_document
# from src.rag.chunking_legal import legal_chunk, semantic_merge_safe
# from src.rag.ingestion import upsert_chunks
# from src.utils.normalize_raw import normalize_raw_to_normalized
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Ingest legal documents into Qdrant (VN law chunking).")
#     parser.add_argument("--input_dir", type=str, default="data/normalized", help="Directory containing .pdf/.txt")
#     parser.add_argument("--glob", type=str, default="*.*", help="Glob pattern, e.g. '*.pdf' or '*.*'")
#
#     # Document-level metadata for this run
#     # parser.add_argument("--law_name", type=str, required=True, help="e.g. NghiDinh07_2026")
#     # parser.add_argument("--law_type", type=str, required=True, help="e.g. NghiDinh / ThongTu / Luat / BoLuat")
#     # parser.add_argument("--year", type=int, required=True, help="e.g. 2026")
#     # parser.add_argument("--source", type=str, default="Local", help="e.g. ChinhPhu / BoNoiVu / Local")
#
#     # Chunk settings
#     parser.add_argument("--max_chars", type=int, default=1600)
#     parser.add_argument("--overlap", type=int, default=120)
#
#     # Semantic merge (safe)
#     parser.add_argument("--semantic_merge", action="store_true")
#     parser.add_argument("--min_chars", type=int, default=350)
#     parser.add_argument("--sim_threshold", type=float, default=0.88)
#     parser.add_argument("--max_merged_chars", type=int, default=1600)
#
#     args = parser.parse_args()
#     normalize_raw_to_normalized()
#     input_dir = Path(args.input_dir)
#     if not input_dir.exists():
#         raise RuntimeError(f"Missing folder: {input_dir}")
#
#     files = sorted(input_dir.glob(args.glob))
#     if not files:
#         raise RuntimeError(f"No files matched: {input_dir}/{args.glob}")
#
#     # meta = {
#     #     "law_name": args.law_name,
#     #     "law_type": args.law_type,
#     #     "year": args.year,
#     #     "source": args.source,
#     # }
#
#     embeddings = get_embedings()
#
#     total = 0
#     for fp in files:
#         if fp.suffix.lower() not in {".pdf", ".txt"}:
#             continue
#         meta = {
#             "law_name": fp.stem,  # VanBanGoc_07.2026.ND-CP
#             "law_type": "NghiDinh",  # tạm hardcode, refine sau
#             "year": 2026,  # có thể parse sau
#             "source": "LocalPDF",  # hoặc ChinhPhu / BoNoiVu
#         }
#         text = load_document(fp)
#         print(f"[DEBUG] {fp.name}: text_len={len(text)}")
#
#         chunks = legal_chunk(text, max_chars=args.max_chars, overlap=args.overlap)
#         print(f"[DEBUG] {fp.name}: chunks={len(chunks)}")
#
#         if not chunks:
#             print(f"[WARN] {fp.name}: no chunks -> SKIP")
#             continue
#
#         if args.semantic_merge:
#             chunks = semantic_merge_safe(
#                 chunks,
#                 embeddings=embeddings,
#                 min_chars=args.min_chars,
#                 sim_threshold=args.sim_threshold,
#                 max_merged_chars=args.max_merged_chars,
#             )
#
#         n = upsert_chunks(chunks, embeddings=embeddings, meta=meta)
#
#         print(f"[OK] {fp.name}: upserted {n} chunks")
#         total += n
#
#     print("DONE. Total chunks:", total)
#
#
# if __name__ == "__main__":
#     main()
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.app.settings import settings
from src.rag.openai_clients import get_embedings
from src.rag.chunking_legal import legal_chunk
from src.rag.ingestion import upsert_chunks
from src.storage.qdrant_store import ensure_collection, get_qdrant_client
from src.utils.loader import load_document
from src.utils.normalize_raw import normalize_raw_to_normalized

try:
    from src.rag.chunking_legal import semantic_merge_safe
except Exception:
    semantic_merge_safe = None


def _slugify(text: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "doc"


def _prefix_chunks(chunks: list[dict], doc_prefix: str) -> list[dict]:
    out = []
    for ch in chunks:
        item = dict(ch)
        md = dict(item.get("metadata", {}) or {})
        if md.get("node_id"):
            md["node_id"] = f"{doc_prefix}::{md['node_id']}"
        if md.get("chunk_id"):
            md["chunk_id"] = f"{doc_prefix}::{md['chunk_id']}"
        item["metadata"] = md
        out.append(item)
    return out


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

    args = parser.parse_args()

    if args.normalize:
        normalize_raw_to_normalized()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Missing folder: {input_dir}")

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No files matched: {input_dir}/{args.glob}")

    client = get_qdrant_client()
    ensure_collection(client)

    embeddings = get_embedings()

    total = 0
    for fp in files:
        if fp.suffix.lower() not in {".pdf", ".txt"}:
            continue

        meta = {
            "law_name": fp.stem,
            "law_type": "Unknown",
            "year": 0,
            "source": "LocalFile",
        }

        text = load_document(fp)
        print(f"[DEBUG] {fp.name}: text_len={len(text)}")

        chunks = legal_chunk(text, max_chars=args.max_chars, overlap=args.overlap)
        print(f"[DEBUG] {fp.name}: chunks={len(chunks)}")

        if not chunks:
            print(f"[WARN] {fp.name}: no chunks -> SKIP")
            continue

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

        doc_prefix = _slugify(fp.stem)
        chunks = _prefix_chunks(chunks, doc_prefix)

        n = upsert_chunks(chunks, embeddings=embeddings, meta=meta)
        print(f"[OK] {fp.name}: upserted {n} chunks")
        total += int(n)

    print("DONE. Total chunks:", total)


if __name__ == "__main__":
    main()
