from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.evaluation.schemas import (
    BENCHMARK_INPUT,
    OPTIONAL_REFERENCE_FIELDS,
    PLACEHOLDER_PATTERNS,
    REQUIRED_BENCHMARK_FIELDS,
    norm_text,
    resolve_benchmark_input,
)


def validate_benchmark(path: str | Path = BENCHMARK_INPUT) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    fp = resolve_benchmark_input(path)
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Failed to read JSON: {exc}"], warnings

    if not isinstance(data, list):
        return False, ["Benchmark input must be a JSON array."], warnings
    if not data:
        errors.append("Benchmark must contain at least 1 sample.")

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            errors.append(f"Sample {idx} must be an object.")
            continue
        for field in REQUIRED_BENCHMARK_FIELDS:
            if field not in item:
                errors.append(f"Sample {idx} missing required field: {field}")
            elif norm_text(item.get(field)) == "":
                errors.append(f"Sample {idx} has empty required field: {field}")
        for field in OPTIONAL_REFERENCE_FIELDS:
            if field not in item:
                warnings.append(f"Sample {idx} missing optional field: {field}; treating as empty.")
        joined = "\n".join(str(item.get(k) or "") for k in item)
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern in joined:
                warnings.append(f"Sample {idx} appears to contain placeholder text: {pattern}")

    return not errors, errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BENCHMARK_INPUT))
    args = parser.parse_args()

    resolved = resolve_benchmark_input(args.input)
    ok, errors, warnings = validate_benchmark(resolved)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not ok:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if str(resolved) != str(Path(args.input)):
        print(f"Using benchmark input: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    sample_count = len(data) if isinstance(data, list) else 0
    print(f"Benchmark valid: {sample_count} samples.")


if __name__ == "__main__":
    main()
