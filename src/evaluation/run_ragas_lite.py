from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.evaluation.ragas_lite_judge import (
    answer_correctness_lite,
    answer_relevancy_lite,
    faithfulness_lite,
    load_cache,
    save_cache,
)
from src.evaluation.rule_metrics import (
    citation_support_lite,
    context_precision_lite,
    context_recall_lite,
    context_sufficiency_lite,
    retrieval_ref_match_lite,
)
from src.evaluation.schemas import JUDGE_CACHE, RAG_OUTPUTS_JSONL, RAGAS_LITE_RESULTS, ensure_parent, read_jsonl

LLM_METRICS: Dict[str, Callable[..., dict]] = {
    "faithfulness": faithfulness_lite,
    "answer_relevancy": answer_relevancy_lite,
    "answer_correctness": answer_correctness_lite,
}

RULE_METRICS = {
    "context_precision": context_precision_lite,
    "context_recall": context_recall_lite,
    "context_sufficiency": context_sufficiency_lite,
    "citation_support": lambda sample: citation_support_lite(str(sample.get("answer") or ""), list(sample.get("context_metadata") or [])),
    "retrieval_ref_match": retrieval_ref_match_lite,
}

METRIC_OUTPUT_NAMES = {
    "faithfulness": "faithfulness_lite",
    "answer_relevancy": "answer_relevancy_lite",
    "answer_correctness": "answer_correctness_lite",
    "context_precision": "context_precision_lite",
    "context_recall": "context_recall_lite",
    "context_sufficiency": "context_sufficiency_lite",
    "citation_support": "citation_support_lite",
    "retrieval_ref_match": "retrieval_ref_match_lite",
}

DEFAULT_METRICS = list(METRIC_OUTPUT_NAMES.keys())


def _parse_metrics(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_METRICS
    metrics = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [metric for metric in metrics if metric not in METRIC_OUTPUT_NAMES]
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}")
    return metrics


def _mean(values: List[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def run_ragas_lite(
    input_path: str = str(RAG_OUTPUTS_JSONL),
    output_path: str = str(RAGAS_LITE_RESULTS),
    *,
    limit: int | None = 25,
    metrics: List[str] | None = None,
    use_cache: bool = False,
    cache_path: str = str(JUDGE_CACHE),
) -> Dict[str, Any]:
    selected = metrics or DEFAULT_METRICS
    samples = read_jsonl(input_path)
    if limit is not None:
        samples = samples[: max(0, int(limit))]

    cache = load_cache(cache_path) if use_cache else {}
    result_samples: List[Dict[str, Any]] = []

    for sample in samples:
        metric_results: Dict[str, Dict[str, Any]] = {}
        for metric in selected:
            out_name = METRIC_OUTPUT_NAMES[metric]
            try:
                if metric in LLM_METRICS:
                    metric_results[out_name] = LLM_METRICS[metric](sample, use_cache=use_cache, cache=cache)
                else:
                    metric_results[out_name] = RULE_METRICS[metric](sample)
            except Exception as exc:
                metric_results[out_name] = {"score": None, "error": str(exc), "reason": f"Metric failed: {exc}"}

        result_samples.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "ground_truth": sample.get("ground_truth", ""),
                "latency_ms": sample.get("latency_ms"),
                "metrics": metric_results,
            }
        )
        if use_cache:
            save_cache(cache, cache_path)

    summary_metrics: Dict[str, Any] = {}
    for metric in selected:
        out_name = METRIC_OUTPUT_NAMES[metric]
        summary_metrics[out_name] = _mean([s["metrics"].get(out_name, {}).get("score") for s in result_samples])

    output = {
        "summary": {
            "num_samples": len(result_samples),
            "metrics": summary_metrics,
        },
        "samples": result_samples,
    }
    ensure_parent(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAG_OUTPUTS_JSONL))
    parser.add_argument("--output", default=str(RAGAS_LITE_RESULTS))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache", default=str(JUDGE_CACHE))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    result = run_ragas_lite(
        args.input,
        args.output,
        limit=args.limit,
        metrics=_parse_metrics(args.metrics),
        use_cache=args.use_cache,
        cache_path=args.cache,
    )
    print(f"Evaluated {result['summary']['num_samples']} samples -> {args.output}")


if __name__ == "__main__":
    main()
