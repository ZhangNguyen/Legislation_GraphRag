from types import SimpleNamespace

from src.rag.local_adequacy_judge import judge_answer_adequacy


def test_local_adequacy_heuristic_flags_generic_refusal_with_evidence(monkeypatch):
    monkeypatch.setattr(
        "src.rag.local_adequacy_judge.settings",
        SimpleNamespace(enable_local_adequacy_judge=True, local_judge_provider="heuristic"),
    )

    judgement = judge_answer_adequacy(
        "Điều 1 quy định gì?",
        "Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này.",
        [{"text": "Điều 1. Bãi bỏ toàn bộ 17 văn bản quy phạm pháp luật."}],
    )

    assert judgement["adequate"] is False
    assert judgement["issue"] == "off_question"
    assert judgement["provider"] == "heuristic"
