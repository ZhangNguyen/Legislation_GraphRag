from src.rag import retrieval_pipeline as rp


def test_heading_list_route_falls_back_when_no_article_bundle(monkeypatch) -> None:
    graph = {"nodes": []}
    doc_candidates = [{"doc_key": "doc_a", "article_bundle_ids": []}]

    fake_candidates = [
        {
            "node_id": "ev_1",
            "doc_key": "doc_a",
            "metadata": {"artifact_type": "evidence"},
            "heading_text": "Mục 1 Yêu cầu chung",
            "self_retrieval_text": "Tập trung bảo đảm an toàn giao thông phục vụ bầu cử.",
        }
    ]

    monkeypatch.setattr(rp, "_build_doc_candidates", lambda doc, graph: fake_candidates)
    monkeypatch.setattr(
        rp,
        "_rank_passages_multifield",
        lambda question, passages, **kwargs: [{**p, "hybrid_score": 0.9} for p in passages],
    )
    monkeypatch.setattr(
        rp,
        "cross_rerank",
        lambda question, passages, top_n, query_profile=None: passages[:top_n],
    )

    pool, final_passages = rp._heading_list_route(
        question="Mục 1 yêu cầu chung là gì?",
        graph=graph,
        doc_candidates=doc_candidates,
        query_profile={"route": "heading_list"},
        question_vec=[0.1, 0.2],
        final_top_k=5,
        cross_top_k=5,
    )

    assert len(pool) == 1
    assert len(final_passages) == 1
    assert final_passages[0]["node_id"] == "ev_1"


def test_heading_list_fallback_filters_recipient_bullets(monkeypatch) -> None:
    graph = {"nodes": []}
    doc_candidates = [{"doc_key": "doc_a", "article_bundle_ids": []}]
    fallback = [
        {
            "node_id": "noise",
            "doc_key": "doc_a",
            "metadata": {"artifact_type": "evidence", "node_type": "bullet", "path_title": "Bullet"},
            "heading_text": "Bullet - Thủ tướng Chính phủ (để b/c);",
            "self_retrieval_text": "- Thủ tướng Chính phủ (để b/c);",
        },
        {
            "node_id": "good",
            "doc_key": "doc_a",
            "metadata": {"artifact_type": "evidence", "node_type": "bullet", "path_title": "Mục 1"},
            "heading_text": "Mục 1 Yêu cầu chung",
            "self_retrieval_text": "Tập trung phối hợp bảo đảm trật tự, an toàn giao thông trong ngày bầu cử.",
        },
    ]
    monkeypatch.setattr(rp, "_build_doc_candidates", lambda doc, graph: fallback)

    seen = {"ids": []}

    def _fake_rank(question, passages, **kwargs):
        seen["ids"] = [p["node_id"] for p in passages]
        return [{**p, "hybrid_score": 1.0} for p in passages]

    monkeypatch.setattr(rp, "_rank_passages_multifield", _fake_rank)
    monkeypatch.setattr(rp, "cross_rerank", lambda question, passages, top_n, query_profile=None: passages[:top_n])

    _, final_passages = rp._heading_list_route(
        question="Mục 1 yêu cầu chung về bảo đảm an toàn giao thông trong ngày bầu cử là gì?",
        graph=graph,
        doc_candidates=doc_candidates,
        query_profile={
            "route": "heading_list",
            "heading_query": "mục 1 yêu cầu chung",
            "focus_terms": ["bảo", "đảm", "an", "toàn", "giao", "thông", "bầu", "cử"],
        },
        question_vec=[0.1, 0.2],
        final_top_k=5,
        cross_top_k=5,
    )

    assert seen["ids"] == ["good"]
    assert [p["node_id"] for p in final_passages] == ["good"]
