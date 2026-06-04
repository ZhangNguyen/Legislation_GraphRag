from src.evaluation.ragas_lite_judge import safe_json_loads
from src.evaluation.rule_metrics import (
    citation_support_lite,
    context_precision_lite,
    context_recall_lite,
    context_sufficiency_lite,
    retrieval_ref_match_lite,
)


def test_context_sufficiency_lite_counts_field_matches():
    sample = {
        "contexts": ["context"],
        "expected_context_refs": [
            {
                "doc_number": "01/2026/TT-NHNN",
                "law_type": "Thông tư",
                "year": "2026",
                "article": "Điều 1",
                "clause": "Khoản 2",
                "point": "",
            }
        ],
        "context_metadata": [
            {
                "doc_number": "01/2026/TT-NHNN",
                "law_type": "Thông tư",
                "year": "2026",
                "article": "Điều 1",
                "clause": "",
            }
        ],
    }

    result = context_sufficiency_lite(sample)

    assert result["score"] == 0.8
    assert "clause" in result["missing_fields"]


def test_retrieval_ref_match_lite_normalizes_when_expected_fields_empty():
    sample = {
        "expected_context_refs": [
            {
                "doc_number": "01/2026/TT-NHNN",
                "law_type": "",
                "year": "",
                "article": "Điều 1",
                "clause": "",
                "point": "",
            }
        ],
        "context_metadata": [{"doc_number": "01/2026/TT-NHNN", "article": ""}],
    }

    result = retrieval_ref_match_lite(sample)

    assert round(result["score"], 4) == round(0.40 / (0.40 + 0.20), 4)
    assert result["matched"]["law_type"] is None


def test_citation_support_lite_flags_unsupported_citation():
    result = citation_support_lite(
        "Theo Điều 6 và Khoản 4, nội dung này được áp dụng.",
        [{"article": "Điều 6"}],
    )

    assert result["score"] == 0.5
    assert "Điều 6" in result["supported_references"]
    assert "Khoản 4" in result["unsupported_references"]


def test_context_precision_and_recall_lite_use_contexts_against_gold():
    sample = {
        "ground_truth": "Bảo hiểm tiền gửi Việt Nam không được cung cấp thông tin cho bên thứ ba.",
        "reference": "Bảo hiểm tiền gửi Việt Nam không được cung cấp thông tin cho bên thứ ba.",
        "source_excerpt": "Bảo hiểm tiền gửi Việt Nam không được cung cấp thông tin cho bên thứ ba.",
        "contexts": [
            "Bảo hiểm tiền gửi Việt Nam không được cung cấp thông tin cho bên thứ ba.",
            "Một đoạn không liên quan về giao thông đường bộ.",
        ],
        "expected_context_refs": [{"doc_number": "01/2026/TT-NHNN", "article": "Điều 4"}],
        "context_metadata": [
            {"doc_number": "01/2026/TT-NHNN", "article": "Điều 4"},
            {"doc_number": "99/2026/TT-BTC", "article": "Điều 9"},
        ],
    }

    precision = context_precision_lite(sample)
    recall = context_recall_lite(sample)

    assert precision["score"] == 0.5
    assert recall["score"] > 0.9


def test_safe_json_loads_parses_markdown_code_block():
    parsed = safe_json_loads(
        """```json
{"score": 0.8, "reason": "ok"}
```"""
    )

    assert parsed["score"] == 0.8
