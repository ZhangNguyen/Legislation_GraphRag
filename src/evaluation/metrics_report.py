from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.schemas import RAGAS_LITE_RESULTS

LOWEST_METRICS = [
    "answer_correctness_lite",
    "faithfulness_lite",
    "context_precision_lite",
    "context_recall_lite",
    "context_sufficiency_lite",
    "retrieval_ref_match_lite",
]


def _score(sample: Dict[str, Any], metric: str) -> float:
    value = sample.get("metrics", {}).get(metric, {}).get("score")
    if isinstance(value, (int, float)):
        return float(value)
    return 999.0


def _reason(sample: Dict[str, Any], metric: str) -> str:
    result = dict(sample.get("metrics", {}).get(metric, {}) or {})
    return str(result.get("reason") or result.get("error") or "")[:180]


def render_report(path: str = str(RAGAS_LITE_RESULTS)) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = dict(data.get("summary") or {})
    metrics = dict(summary.get("metrics") or {})
    samples = [dict(x) for x in list(data.get("samples") or [])]

    lines: List[str] = []
    lines.append(f"{'Metric':30} Mean")
    for name, value in metrics.items():
        rendered = "n/a" if value is None else f"{float(value):.2f}"
        lines.append(f"{name:30} {rendered}")

    latencies = [float(s.get("latency_ms")) for s in samples if isinstance(s.get("latency_ms"), (int, float))]
    if latencies:
        lines.append("")
        lines.append(f"Average latency_ms: {sum(latencies) / len(latencies):.1f}")

    no_context = [s for s in samples if not s.get("metrics", {}).get("context_sufficiency_lite") and not s.get("answer")]
    empty_contexts = [
        s
        for s in samples
        if s.get("metrics", {}).get("context_sufficiency_lite", {}).get("score") == 0.0
    ]
    if empty_contexts:
        lines.append("")
        lines.append(f"WARNING: {len(empty_contexts)} samples have insufficient or empty context.")

    for metric in LOWEST_METRICS:
        if metric not in metrics:
            continue
        ranked = sorted(samples, key=lambda sample: _score(sample, metric))[:5]
        lines.append("")
        lines.append(f"Lowest {metric}:")
        for sample in ranked:
            score = sample.get("metrics", {}).get(metric, {}).get("score")
            rendered = "n/a" if score is None else f"{float(score):.2f}"
            suffix = ""
            if metric == "retrieval_ref_match_lite":
                suffix = f", question={str(sample.get('question') or '')[:80]}"
            reason = _reason(sample, metric)
            lines.append(f"- {sample.get('id')}: score={rendered}{suffix}, reason={reason}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAGAS_LITE_RESULTS))
    args = parser.parse_args()
    print(render_report(args.input))


if __name__ == "__main__":
    main()
