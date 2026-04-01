from src.rag.retrieval_pipeline import _context_text_for_node


def test_context_text_contains_parent_child_sentences() -> None:
    graph = {
        "nodes": [
            {
                "node_id": "bundle_1",
                "text": "Điều này quy định trách nhiệm chung. UBND phối hợp thực hiện.",
                "metadata": {"path_title": "Điều 1"},
            },
            {
                "node_id": "child_1",
                "text": "Sở Công Thương chủ trì tham mưu.",
                "metadata": {"path_title": "Khoản 1"},
            },
            {
                "node_id": "ev_1",
                "text": "a) Chỉ đạo bảo đảm cung ứng xăng dầu.",
                "metadata": {
                    "article": "Điều 1",
                    "children_ids": ["child_1"],
                },
            },
        ]
    }
    doc_info = {
        "article_to_bundle_id": {"Điều 1": "bundle_1"},
    }
    node = graph["nodes"][2]
    context = _context_text_for_node(node, doc_info, graph)
    assert "[Ngữ cảnh cha-con]" in context
    assert "Điều này quy định trách nhiệm chung." in context
    assert "Sở Công Thương chủ trì tham mưu." in context
