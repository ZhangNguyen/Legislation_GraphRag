from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.evaluation.collect_rag_outputs import collect_outputs
from src.evaluation.metrics_report import render_report
from src.evaluation.prepare_benchmark import prepare_benchmark
from src.evaluation.run_ragas_lite import DEFAULT_METRICS, _parse_metrics, run_ragas_lite
from src.evaluation.schemas import (
    BENCHMARK_INPUT,
    BENCHMARK_JSONL,
    JUDGE_CACHE,
    RAG_OUTPUTS_JSONL,
    RAGAS_LITE_RESULTS,
    resolve_benchmark_input,
)
from src.evaluation.validate_benchmark import validate_benchmark


def _print_validation(warnings: list[str], errors: list[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")


def run_all(
    *,
    input_path: str = str(BENCHMARK_INPUT),
    benchmark_jsonl_path: str = str(BENCHMARK_JSONL),
    rag_outputs_path: str = str(RAG_OUTPUTS_JSONL),
    results_path: str = str(RAGAS_LITE_RESULTS),
    top_k: int = 5,
    limit: int | None = None,
    metrics: str = ",".join(DEFAULT_METRICS),
    use_cache: bool = True,
    cache_path: str = str(JUDGE_CACHE),
) -> Dict[str, Any]:
    resolved_input = resolve_benchmark_input(input_path)

    print("[1/5] Validate benchmark")
    ok, errors, warnings = validate_benchmark(resolved_input)
    _print_validation(warnings, errors)
    if not ok:
        raise SystemExit(1)
    print(f"Benchmark valid: {resolved_input}")

    print("[2/5] Prepare benchmark JSONL")
    samples = prepare_benchmark(resolved_input, benchmark_jsonl_path)
    print(f"Prepared {len(samples)} samples -> {benchmark_jsonl_path}")

    print("[3/5] Collect RAG outputs")
    rows = collect_outputs(
        benchmark_jsonl_path,
        rag_outputs_path,
        top_k=top_k,
        limit=limit,
    )
    print(f"Collected {len(rows)} RAG outputs -> {rag_outputs_path}")

    print("[4/5] Run RAGAS-lite")
    result = run_ragas_lite(
        rag_outputs_path,
        results_path,
        limit=None,
        metrics=_parse_metrics(metrics),
        use_cache=use_cache,
        cache_path=cache_path,
    )
    print(f"Evaluated {result['summary']['num_samples']} samples -> {results_path}")

    print("[5/5] Report")
    report = render_report(results_path)
    print(report)

    return {
        "benchmark_input": str(Path(resolved_input)),
        "benchmark_jsonl": benchmark_jsonl_path,
        "rag_outputs": rag_outputs_path,
        "results": results_path,
        "num_prepared": len(samples),
        "num_collected": len(rows),
        "num_evaluated": result["summary"]["num_samples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BENCHMARK_INPUT))
    parser.add_argument("--benchmark-jsonl", default=str(BENCHMARK_JSONL))
    parser.add_argument("--rag-outputs", default=str(RAG_OUTPUTS_JSONL))
    parser.add_argument("--output", default=str(RAGAS_LITE_RESULTS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--cache", default=str(JUDGE_CACHE))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_all(
        input_path=args.input,
        benchmark_jsonl_path=args.benchmark_jsonl,
        rag_outputs_path=args.rag_outputs,
        results_path=args.output,
        top_k=args.top_k,
        limit=args.limit,
        metrics=args.metrics,
        use_cache=not args.no_cache,
        cache_path=args.cache,
    )


if __name__ == "__main__":
    main()
