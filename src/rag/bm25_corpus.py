from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from src.rag.chunking_legal import legal_chunk
from src.utils.loader import load_document

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "") if tok.strip()]


def iter_chunk_texts_from_normalized(
    input_dir: str = "data/normalized",
    glob_pattern: str = "*.*",
    max_chars: int = 1600,
    overlap: int = 120,
) -> List[str]:
    texts: List[str] = []
    base = Path(input_dir)

    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    for fp in sorted(base.glob(glob_pattern)):
        if fp.suffix.lower() not in {".pdf", ".txt"}:
            continue

        raw_text = load_document(fp)
        chunks = legal_chunk(raw_text, max_chars=max_chars, overlap=overlap)

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = str(chunk.get("text", "")).strip()
            else:
                text = str(chunk).strip()

            if text:
                texts.append(text)

    return texts


def build_bm25_stats_from_texts(texts: List[str]) -> Dict[str, Any]:
    tokenized_docs: List[List[str]] = []
    df: Dict[str, int] = {}

    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            continue

        tokenized_docs.append(tokens)
        seen = set(tokens)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1

    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return {
            "N": 0,
            "avgdl": 0.0,
            "df": {},
            "idf": {},
        }

    avgdl = sum(len(doc) for doc in tokenized_docs) / n_docs

    idf: Dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    return {
        "N": n_docs,
        "avgdl": avgdl,
        "df": df,
        "idf": idf,
    }


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
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing BM25 stats file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))