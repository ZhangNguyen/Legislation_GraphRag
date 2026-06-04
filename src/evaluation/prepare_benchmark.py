from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.schemas import BENCHMARK_INPUT, BENCHMARK_JSONL, expected_ref_from_sample, resolve_benchmark_input, write_jsonl


def prepare_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(data, start=1):
        sample = dict(raw)
        sample_id = str(sample.get("id") or f"q{idx:03d}")
        out.append(
            {
                "id": sample_id,
                "question": str(sample.get("question") or ""),
                "ground_truth": str(sample.get("ground_truth") or ""),
                "reference": str(sample.get("reference") or sample.get("ground_truth") or ""),
                "expected_context_refs": [expected_ref_from_sample(sample)],
                "source_excerpt": str(sample.get("source_excerpt") or ""),
            }
        )
    return out


def prepare_benchmark(input_path: str | Path = BENCHMARK_INPUT, output_path: str | Path = BENCHMARK_JSONL) -> List[Dict[str, Any]]:
    resolved_input = resolve_benchmark_input(input_path)
    data = json.loads(resolved_input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark input must be a JSON array.")
    samples = prepare_samples([dict(item) for item in data])
    write_jsonl(output_path, samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BENCHMARK_INPUT))
    parser.add_argument("--output", default=str(BENCHMARK_JSONL))
    args = parser.parse_args()

    resolved_input = resolve_benchmark_input(args.input)
    samples = prepare_benchmark(resolved_input, args.output)
    if str(resolved_input) != str(Path(args.input)):
        print(f"Using benchmark input: {resolved_input}")
    print(f"Prepared {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
