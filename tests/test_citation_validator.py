from src.rag.citation_validator import validate_citations


def test_citation_validator_flags_unsupported_article():
    result = validate_citations(
        "Theo Điều 6, người vi phạm bị xử lý.",
        [{"metadata": {"article": "Điều 5"}}],
    )

    assert result["valid"] is False
    assert result["unsupported_references"]


def test_citation_validator_does_not_treat_quan_diem_as_point_reference():
    result = validate_citations(
        (
            "Quan \u0111i\u1ec3m x\u00e2y d\u1ef1ng \u0111\u1ec1 \u00e1n "
            "g\u1ed3m c\u00e1c n\u1ed9i dung tr\u00ean; "
            "\u0111i\u1ec3m n\u00e0y \u0111\u00e3 \u0111\u01b0\u1ee3c n\u00eau r\u00f5."
        ),
        [{"metadata": {"path_title": "\u0110\u1ec0 \u00c1N > I. QUAN \u0110I\u1ec2M X\u00c2Y D\u1ef0NG \u0110\u1ec0 \u00c1N"}}],
    )

    assert result["valid"] is True
    assert result["unsupported_references"] == []


def test_citation_validator_allows_doc_numbers_present_in_source_text():
    result = validate_citations(
        "Bãi bỏ Quyết định số 61/2004/QĐ-UB và Quyết định số 62/2004/QĐ-UB.",
        [
            {"metadata": {"doc_number": "18/2026/QĐ-UBND"}, "text": "1. Quyết định số 61/2004/QĐ-UB ngày 20 tháng 8 năm 2004."},
            {"metadata": {"doc_number": "18/2026/QĐ-UBND"}, "text": "2. Quyết định số 62/2004/QĐ-UB ngày 20 tháng 8 năm 2004."},
        ],
    )

    assert result["valid"] is True
    assert result["unsupported_references"] == []
