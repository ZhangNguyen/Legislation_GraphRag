from src.rag.context_grader import grade_context
from src.rag.retrieval_pipeline_simple import analyze_query


def test_context_grader_explicit_reference_requires_matching_metadata():
    question = "Điểm a khoản 3 Điều 6 quy định gì?"
    profile = analyze_query(question)
    wrong_passages = [
        {
            "text": "Một quy định liên quan.",
            "metadata": {"article": "Điều 5", "clause": "Khoản 3", "point": "Điểm a"},
        }
    ]

    assert grade_context(question, profile, wrong_passages)["status"] == "insufficient"

    right_passages = [
        {
            "text": "Điểm a khoản 3 Điều 6 quy định về việc không chấp hành tín hiệu giao thông.",
            "metadata": {"article": "Điều 6", "clause": "Khoản 3", "point": "Điểm a"},
        }
    ]

    assert grade_context(question, profile, right_passages)["status"] == "sufficient"


def test_context_grader_needs_reasoning_for_fault_question_with_relevant_rules():
    question = "Xe A vượt đèn đỏ va chạm xe B, ai có lỗi?"
    profile = analyze_query(question)
    passages = [
        {
            "text": "Người điều khiển phương tiện phải chấp hành tín hiệu giao thông; không được vượt đèn đỏ.",
            "metadata": {"article": "Điều 6"},
        }
    ]

    result = grade_context(question, profile, passages)

    assert result["status"] == "needs_reasoning"
    assert result["allow_reasoning"] is True
