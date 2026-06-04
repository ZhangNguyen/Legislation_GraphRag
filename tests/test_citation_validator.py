from src.rag.citation_validator import validate_citations


def test_citation_validator_flags_unsupported_article():
    result = validate_citations(
        "Theo Điều 6, người vi phạm bị xử lý.",
        [{"metadata": {"article": "Điều 5"}}],
    )

    assert result["valid"] is False
    assert result["unsupported_references"]
