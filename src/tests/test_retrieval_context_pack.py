from src.rag.retrieval_pipeline import _context_text_for_node, _should_force_heading_route


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


def test_force_heading_route_when_heading_overlap_is_high() -> None:
    graph = {
        "nodes": [
            {
                "node_id": "bundle_1",
                "text": "Quy định về cấp giấy phép và điều kiện thực hiện.",
                "metadata": {"artifact_type": "article_bundle", "path_title": "Mục 1 Cấp giấy phép"},
            }
        ],
        "edges": [],
    }
    doc_candidates = [
        {
            "doc_key": "doc_1",
            "law_name": "VB test",
            "law_type": "Thông tư",
            "source": "LocalFile",
            "rrf_score": 1.0,
            "article_bundle_ids": ["bundle_1"],
            "evidence_ids": [],
            "article_to_bundle_id": {},
        }
    ]
    profile = {"route": "factoid", "filters": {}}
    forced = _should_force_heading_route(
        question="Quy định cấp giấy phép là gì?",
        graph=graph,
        doc_candidates=doc_candidates,
        query_profile=profile,
        question_vec=[0.1, 0.2, 0.3],
    )
    assert forced is True
