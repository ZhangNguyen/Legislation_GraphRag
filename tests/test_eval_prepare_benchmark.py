import json
from pathlib import Path

from src.evaluation.prepare_benchmark import prepare_benchmark


def test_prepare_benchmark_converts_json_array_to_jsonl(tmp_path: Path):
    input_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "benchmark.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {
                    "question": "Điều 1 quy định gì?",
                    "ground_truth": "Điều 1 quy định phạm vi.",
                    "reference": "Điều 1 quy định phạm vi.",
                    "expected_doc_number": "01/2026/TT-NHNN",
                    "expected_law_type": "Thông tư",
                    "expected_year": "2026",
                    "expected_article": "Điều 1",
                    "expected_clause": "",
                    "expected_point": "",
                    "source_excerpt": "Điều 1. Phạm vi điều chỉnh",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    samples = prepare_benchmark(input_path, output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert len(samples) == 1
    assert payload["id"] == "q001"
    assert payload["expected_context_refs"][0]["doc_number"] == "01/2026/TT-NHNN"
    assert payload["expected_context_refs"][0]["article"] == "Điều 1"
