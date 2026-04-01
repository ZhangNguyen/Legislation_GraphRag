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


def test_section_reference_prefers_heading_list_and_extracts_section() -> None:
    profile = infer_query_profile("Mục 1 yêu cầu chung của chính phủ về bảo đảm an toàn giao thông là gì?")
    assert profile["route"] == "heading_list"
    assert profile["section_query"] == "Mục 1"
    assert "yêu cầu chung" in profile["section_title_hint"]


def test_implicit_structure_query_without_dieu_khoan_still_routes_heading_list() -> None:
    profile = infer_query_profile("Yêu cầu chung của Chính phủ về bảo đảm an toàn giao thông trong ngày bầu cử là gì?")
    assert profile["route"] == "heading_list"
    assert "giao" in profile["focus_terms"]
    assert "toàn" in profile["focus_terms"]
