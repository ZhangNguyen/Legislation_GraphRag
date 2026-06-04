from src.rag.retrieval_pipeline_simple import rrf_fusion


def test_rrf_prefers_item_present_in_both_rank_lists():
    dense = [
        {"id": "dense-only", "rank": 1, "text": "dense"},
        {"id": "shared", "rank": 2, "text": "shared"},
    ]
    bm25 = [
        {"id": "bm25-only", "rank": 1, "text": "bm25"},
        {"id": "shared", "rank": 2, "text": "shared"},
    ]

    fused = rrf_fusion([dense, bm25], rrf_k=60, top_k=3)
    ids = [item["id"] for item in fused]

    assert ids[0] == "shared"
