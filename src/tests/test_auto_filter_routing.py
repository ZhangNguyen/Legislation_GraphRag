from src.rag.auto_filter import infer_query_profile


def test_responsibility_question_prefers_heading_list_route() -> None:
    profile = infer_query_profile(
        "ủy ban nhân dân các tỉnh thành phố làm gì để bảo đảm an ninh năng lượng trước cuộc xung đột trung đông"
    )
    assert profile["route"] == "heading_list"
    assert profile["wants_list_answer"] is True
    assert profile["asks_responsibility"] is True


def test_generic_condition_keywords_do_not_overtrigger_route() -> None:
    profile = infer_query_profile("Doanh nghiệp được hỗ trợ như thế nào khi gặp khó khăn?")
    assert profile["route"] != "condition_circumstance"
