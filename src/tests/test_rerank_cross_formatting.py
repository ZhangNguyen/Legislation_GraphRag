from src.rag.rerank_cross import (
    _build_rerank_text,
    _truncate_question_words,
)


def test_truncate_question_words_to_30() -> None:
    question = " ".join(f"t{i}" for i in range(40))
    truncated = _truncate_question_words(question, max_words=30)
    assert len(truncated.split()) == 30


def test_build_rerank_text_injects_headers_only_when_missing() -> None:
    passage = {
        "text": "Nội dung mẫu",
        "metadata": {
            "official_title": "Chỉ thị mẫu",
            "article": "Điều 1",
            "clause": "Khoản 2",
        },
    }
    rerank_text = _build_rerank_text(passage)
    assert "[Văn bản] Chỉ thị mẫu" in rerank_text
    assert "[Vị trí]" in rerank_text

    passage_has_headers = {
        "rerank_text_short": "[Văn bản] Chỉ thị mẫu\n\n[Vị trí] Điều 1 | artifact=evidence\n\nNội dung mẫu",
        "metadata": {"official_title": "Chỉ thị mẫu", "article": "Điều 1"},
    }
    rerank_text_2 = _build_rerank_text(passage_has_headers)
    assert rerank_text_2.count("[Văn bản]") == 1
