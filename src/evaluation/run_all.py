from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.evaluation.collect_rag_outputs import collect_outputs
from src.evaluation.metrics_report import render_report
from src.evaluation.prepare_benchmark import prepare_benchmark
from src.evaluation.qdrant_preflight import check_qdrant_ready
from src.evaluation.run_ragas import DEFAULT_METRICS, _parse_metrics, run_ragas
from src.evaluation.schemas import (
    BENCHMARK_INPUT,
    BENCHMARK_JSONL,
    JUDGE_CACHE,
    RAG_OUTPUTS_JSONL,
    RAGAS_RESULTS,
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
    results_path: str = str(RAGAS_RESULTS),
    top_k: int = 5,
    limit: int | None = None,
    metrics: str = ",".join(DEFAULT_METRICS),
    use_cache: bool = True,
    cache_path: str = str(JUDGE_CACHE),
    qdrant_preflight: bool = True,
    preflight_only: bool = False,
) -> Dict[str, Any]:
    resolved_input = resolve_benchmark_input(input_path)

    step_count = 6 if qdrant_preflight else 5
    if qdrant_preflight:
        print(f"[1/{step_count}] Qdrant preflight")
        qdrant_info = check_qdrant_ready()
        print(
            "Qdrant ready: "
            f"url={qdrant_info.url}, "
            f"collection={qdrant_info.collection}, "
            f"status={qdrant_info.status}, "
            f"points_count={qdrant_info.points_count}"
        )
    else:
        qdrant_info = None

    if preflight_only:
        if not qdrant_info:
            raise RuntimeError("--preflight-only requires Qdrant preflight to be enabled.")
        return {
            "benchmark_input": str(Path(resolved_input)),
            "qdrant": qdrant_info.as_dict(),
        }

    print(f"[{2 if qdrant_preflight else 1}/{step_count}] Validate benchmark")
    ok, errors, warnings = validate_benchmark(resolved_input)
    _print_validation(warnings, errors)
    if not ok:
        raise SystemExit(1)
    print(f"Benchmark valid: {resolved_input}")

    print(f"[{3 if qdrant_preflight else 2}/{step_count}] Prepare benchmark JSONL")
    samples = prepare_benchmark(resolved_input, benchmark_jsonl_path)
    print(f"Prepared {len(samples)} samples -> {benchmark_jsonl_path}")

    print(f"[{4 if qdrant_preflight else 3}/{step_count}] Collect RAG outputs")
    rows = collect_outputs(
        benchmark_jsonl_path,
        rag_outputs_path,
        top_k=top_k,
        limit=limit,
    )
    print(f"Collected {len(rows)} RAG outputs -> {rag_outputs_path}")

    del use_cache, cache_path

    print(f"[{5 if qdrant_preflight else 4}/{step_count}] Run standard RAGAS")
    result = run_ragas(
        rag_outputs_path,
        results_path,
        limit=None,
        metrics=_parse_metrics(metrics),
    )
    print(f"Evaluated {result['summary']['num_samples']} samples -> {results_path}")

    print(f"[{6 if qdrant_preflight else 5}/{step_count}] Report")
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
        "qdrant": qdrant_info.as_dict() if qdrant_info else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BENCHMARK_INPUT))
    parser.add_argument("--benchmark-jsonl", default=str(BENCHMARK_JSONL))
    parser.add_argument("--rag-outputs", default=str(RAG_OUTPUTS_JSONL))
    parser.add_argument("--output", default=str(RAGAS_RESULTS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--cache", default=str(JUDGE_CACHE), help=argparse.SUPPRESS)
    parser.add_argument("--no-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-qdrant-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
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
        qdrant_preflight=not args.skip_qdrant_preflight,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
