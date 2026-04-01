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
