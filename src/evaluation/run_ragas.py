from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.schemas import RAG_OUTPUTS_JSONL, RAGAS_RESULTS, ensure_parent, read_jsonl
from src.rag.openai_clients import get_embedings, get_llm


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _install_ragas_vertexai_import_shim() -> None:
    """RAGAS imports optional VertexAI classes at module import time on some versions."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(module_name)

    class _UnavailableVertexAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("VertexAI is not installed and is not used by this OpenAI RAGAS runner.")

    module.ChatVertexAI = _UnavailableVertexAI
    sys.modules[module_name] = module


def _parse_metrics(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_METRICS
    metrics = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [metric for metric in metrics if metric not in DEFAULT_METRICS]
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {', '.join(unknown)}")
    return metrics


def _load_ragas_metric(name: str) -> Any:
    _install_ragas_vertexai_import_shim()
    try:
        import ragas.metrics as metrics_module
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ragas'. Install standard RAGAS first, for example: "
            "pip install ragas datasets"
        ) from exc

    try:
        return getattr(metrics_module, name)
    except AttributeError as exc:
        raise ValueError(f"RAGAS metric is not available in installed version: {name}") from exc


def _dataset_from_samples(samples: List[Dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with standard RAGAS: pip install ragas datasets"
        ) from exc

    rows = []
    for sample in samples:
        rows.append(
            {
                "question": str(sample.get("question") or ""),
                "answer": str(sample.get("answer") or ""),
                "contexts": [str(ctx) for ctx in list(sample.get("contexts") or [])],
                "ground_truth": str(sample.get("ground_truth") or sample.get("reference") or ""),
            }
        )
    return Dataset.from_list(rows)


def _ragas_score_to_dict(score: Any) -> Dict[str, Any]:
    if hasattr(score, "to_pandas"):
        from pandas.api.types import is_numeric_dtype

        frame = score.to_pandas()
        records = frame.to_dict(orient="records")
        means = {
            col: float(frame[col].mean())
            for col in frame.columns
            if col not in {"question", "answer", "contexts", "ground_truth"} and is_numeric_dtype(frame[col])
        }
        return {"records": records, "summary_metrics": means}

    if isinstance(score, dict):
        records = score.get("records") or score.get("samples") or []
        metrics = {
            key: float(value)
            for key, value in score.items()
            if isinstance(value, (int, float))
        }
        return {"records": records, "summary_metrics": metrics}

    raise TypeError(f"Unsupported RAGAS result type: {type(score)!r}")


def run_ragas(
    input_path: str = str(RAG_OUTPUTS_JSONL),
    output_path: str = str(RAGAS_RESULTS),
    *,
    limit: int | None = 25,
    metrics: List[str] | None = None,
) -> Dict[str, Any]:
    selected = metrics or DEFAULT_METRICS
    samples = read_jsonl(input_path)
    if limit is not None:
        samples = samples[: max(0, int(limit))]

    try:
        _install_ragas_vertexai_import_shim()
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ragas'. This runner uses standard RAGAS, not RAGAS-lite."
        ) from exc

    dataset = _dataset_from_samples(samples)
    metric_objects = [_load_ragas_metric(metric) for metric in selected]

    score = evaluate(
        dataset,
        metrics=metric_objects,
        llm=get_llm(),
        embeddings=get_embedings(),
    )
    converted = _ragas_score_to_dict(score)
    records = list(converted["records"])

    result_samples: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row = dict(records[idx]) if idx < len(records) and isinstance(records[idx], dict) else {}
        metric_results: Dict[str, Dict[str, Any]] = {}
        for metric in selected:
            value = row.get(metric)
            metric_results[metric] = {
                "score": float(value) if isinstance(value, (int, float)) else None,
            }

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

    output = {
        "summary": {
            "num_samples": len(result_samples),
            "metrics": converted["summary_metrics"],
        },
        "samples": result_samples,
    }
    ensure_parent(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAG_OUTPUTS_JSONL))
    parser.add_argument("--output", default=str(RAGAS_RESULTS))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    result = run_ragas(
        args.input,
        args.output,
        limit=args.limit,
        metrics=_parse_metrics(args.metrics),
    )
    print(f"Evaluated {result['summary']['num_samples']} samples with standard RAGAS -> {args.output}")


if __name__ == "__main__":
    main()
