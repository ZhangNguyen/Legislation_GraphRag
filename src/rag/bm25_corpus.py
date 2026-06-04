from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from src.rag.chunking_legal import legal_chunk
from src.rag.hybrid import tokenize_vi
from src.utils.loader import load_document


ARTIFACTS_FOR_BM25 = {"evidence", "article_bundle", "section_bundle", "doc_sketch"}


def iter_chunk_texts_from_normalized(
    input_dir: str = "data/normalized",
    glob_pattern: str = "*.*",
    max_chars: int = 1600,
    overlap: int = 120,
) -> List[str]:
    texts: List[str] = []
    base = Path(input_dir)

    if not base.exists():
        return []

    for fp in sorted(base.glob(glob_pattern)):
        if fp.suffix.lower() not in {".pdf", ".txt"}:
            continue

        raw_text = load_document(fp)
        chunks = legal_chunk(raw_text, max_chars=max_chars, overlap=overlap, fallback_doc_name=fp.stem)

        for chunk in chunks:
            if not isinstance(chunk, dict):
                text = str(chunk).strip()
            else:
                md = dict(chunk.get("metadata") or {})
                artifact = str(md.get("artifact_type") or "evidence")
                if artifact not in ARTIFACTS_FOR_BM25:
                    continue
                text = str(chunk.get("retrieval_text") or chunk.get("text") or "").strip()
            if text:
                texts.append(text)

    return texts


def build_bm25_stats_from_texts(texts: List[str]) -> Dict[str, Any]:
    tokenized_docs: List[List[str]] = []
    df: Dict[str, int] = {}

    for text in texts:
        tokens = tokenize_vi(text)
        if not tokens:
            continue
        tokenized_docs.append(tokens)
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return {"N": 0, "avgdl": 0.0, "df": {}, "idf": {}}

    avgdl = sum(len(doc) for doc in tokenized_docs) / max(n_docs, 1)

    idf: Dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    return {"N": n_docs, "avgdl": float(avgdl), "df": df, "idf": idf}


def build_bm25_stats_from_normalized(
    input_dir: str = "data/normalized",
    glob_pattern: str = "*.*",
    max_chars: int = 1600,
    overlap: int = 120,
) -> Dict[str, Any]:
    texts = iter_chunk_texts_from_normalized(
        input_dir=input_dir,
        glob_pattern=glob_pattern,
        max_chars=max_chars,
        overlap=overlap,
    )
    return build_bm25_stats_from_texts(texts)


def save_bm25_stats(stats: Dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")


def load_bm25_stats(path: str) -> Dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        raise RuntimeError(f"Missing BM25 stats file: {fp}")
    return json.loads(fp.read_text(encoding="utf-8"))
