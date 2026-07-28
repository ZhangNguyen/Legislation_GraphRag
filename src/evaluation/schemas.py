from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

BENCHMARK_INPUT = Path("data/eval/legal_rag_benchmark_25.json")
BENCHMARK_INPUT_FALLBACKS = [
    Path("outputs/benchmark/ragas_gold_25_manual.json"),
    Path("outputs/benchmark/ragas_gold_25.json"),
]
BENCHMARK_JSONL = Path("data/eval/legal_benchmark_25.jsonl")
RAG_OUTPUTS_JSONL = Path("data/eval/rag_outputs.jsonl")
RAGAS_RESULTS = Path("data/eval/ragas_results.json")
RAGAS_LITE_RESULTS = Path("data/eval/ragas_lite_results.json")
JUDGE_CACHE = Path("data/eval/judge_cache.json")

REQUIRED_BENCHMARK_FIELDS = [
    "question",
    "ground_truth",
    "reference",
    "expected_doc_number",
    "expected_law_type",
    "expected_year",
    "source_excerpt",
]

OPTIONAL_REFERENCE_FIELDS = [
    "expected_article",
    "expected_clause",
    "expected_point",
]

REFERENCE_FIELDS = [
    "doc_number",
    "law_type",
    "year",
    "article",
    "clause",
    "point",
]

PLACEHOLDER_PATTERNS = [
    "Câu hỏi benchmark số",
    "Đáp án chuẩn cho câu hỏi benchmark số",
    "Trích đoạn nguồn cho câu hỏi benchmark số",
]


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def resolve_benchmark_input(path: str | Path = BENCHMARK_INPUT) -> Path:
    requested = Path(path)
    if requested.exists():
        return requested
    if requested == BENCHMARK_INPUT:
        for fallback in BENCHMARK_INPUT_FALLBACKS:
            if fallback.exists():
                return fallback
    return requested


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def norm_key(value: Any) -> str:
    return norm_text(value).lower()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(dict(__import__("json").loads(line)))
    return items


def write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> None:
    import json

    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def expected_ref_from_sample(sample: Dict[str, Any]) -> Dict[str, str]:
    return {
        "doc_number": norm_text(sample.get("expected_doc_number")),
        "law_type": norm_text(sample.get("expected_law_type")),
        "year": norm_text(sample.get("expected_year")),
        "article": norm_text(sample.get("expected_article")),
        "clause": norm_text(sample.get("expected_clause")),
        "point": norm_text(sample.get("expected_point")),
    }
