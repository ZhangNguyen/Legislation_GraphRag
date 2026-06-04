from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from src.app.runtime import ensure_runtime_graph
from src.app.settings import settings
from src.evaluation.schemas import BENCHMARK_JSONL, RAG_OUTPUTS_JSONL, read_jsonl, write_jsonl
from src.evaluation.prepare_benchmark import prepare_benchmark
from src.rag.qa_engine import answer_with_rag


def _retrieve(question: str, graph: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    pipeline = str(getattr(settings, "retrieval_pipeline", "simple_rrf") or "simple_rrf").lower()
    if pipeline in {"simple", "simple_rrf"}:
        from src.rag.context_grader import grade_context
        from src.rag.retrieval_pipeline_simple import retrieve_with_graph_simple

        result = retrieve_with_graph_simple(question, graph, final_top_k=top_k)
        context_grade = grade_context(question, dict(result.get("query_profile") or {}), list(result.get("passages") or []))
        result["context_grade"] = context_grade
        result["insufficient_context"] = context_grade.get("status") == "insufficient"
        return result

    from src.rag.retrieval_pipeline import retrieve_with_graph

    return retrieve_with_graph(question=question, graph=graph, final_top_k=top_k)


def _context_text(passage: Dict[str, Any]) -> str:
    for key in ("retrieval_text", "local_text", "text", "snippet"):
        value = passage.get(key)
        if value is not None and str(value).strip():
            return " ".join(str(value).split()).strip()
    return ""


def _metadata_for_context(passage: Dict[str, Any]) -> Dict[str, str]:
    md = dict(passage.get("metadata") or {})
    return {
        "chunk_id": str(passage.get("chunk_id") or passage.get("node_id") or md.get("chunk_id") or md.get("node_id") or ""),
        "doc_number": str(md.get("doc_number") or ""),
        "law_type": str(md.get("law_type") or md.get("doc_type") or ""),
        "year": str(md.get("year") or ""),
        "article": str(md.get("article") or ""),
        "clause": str(md.get("clause") or ""),
        "point": str(md.get("point") or ""),
        "source": str(md.get("source") or md.get("issuing_agency") or ""),
    }


def collect_outputs(input_path: str = str(BENCHMARK_JSONL), output_path: str = str(RAG_OUTPUTS_JSONL), top_k: int = 5, limit: int | None = None) -> List[Dict[str, Any]]:
    if not Path(input_path).exists() and Path(input_path) == BENCHMARK_JSONL:
        prepare_benchmark(output_path=BENCHMARK_JSONL)
    samples = read_jsonl(input_path)
    if limit is not None:
        samples = samples[: max(0, int(limit))]
    graph = ensure_runtime_graph()

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question") or "")
        t0 = time.perf_counter()
        retrieval_result = _retrieve(question, graph, top_k=top_k)
        response = answer_with_rag(question, retrieval_result)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        contexts: List[str] = []
        context_metadata: List[Dict[str, str]] = []
        seen = set()
        for passage in list(retrieval_result.get("passages") or [])[:top_k]:
            text = _context_text(dict(passage))
            if not text or text in seen:
                continue
            seen.add(text)
            contexts.append(text)
            context_metadata.append(_metadata_for_context(dict(passage)))

        rows.append(
            {
                "id": sample.get("id"),
                "question": question,
                "answer": response.answer,
                "ground_truth": sample.get("ground_truth", ""),
                "reference": sample.get("reference", sample.get("ground_truth", "")),
                "contexts": contexts,
                "context_metadata": context_metadata,
                "expected_context_refs": list(sample.get("expected_context_refs") or []),
                "source_excerpt": sample.get("source_excerpt", ""),
                "latency_ms": latency_ms,
            }
        )

    write_jsonl(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BENCHMARK_JSONL))
    parser.add_argument("--output", default=str(RAG_OUTPUTS_JSONL))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = collect_outputs(args.input, args.output, top_k=args.top_k, limit=args.limit)
    print(f"Collected {len(rows)} RAG outputs -> {args.output}")


if __name__ == "__main__":
    main()
