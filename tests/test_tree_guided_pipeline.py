from src.rag.auto_filter import infer_query_profile
from src.rag.tree_guided_pipeline import (
    _full_relation_node_ids,
    _infer_document_filter_from_question,
    _iter_summary_candidates,
    retrieve_tree_guided,
)


def _graph():
    article_id = "doc1::article::2"
    clause_id = "doc1::clause::2::1"
    point_id = "doc1::point::2::1::a"
    nodes = [
        {
            "node_id": article_id,
            "node_type": "article",
            "text": "Dieu 2. Nguyen tac chung.",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": article_id,
                "chunk_id": article_id,
                "node_type": "article",
                "children_ids": [clause_id],
                "article": "Dieu 2",
                "path_title": "Dieu 2",
                "official_title": "Test law",
                "doc_number": "01/2026/TT-TEST",
            },
        },
        {
            "node_id": clause_id,
            "node_type": "clause",
            "text": "Khoan 1 quy dinh noi dung rieng cua khoan.",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": clause_id,
                "chunk_id": clause_id,
                "node_type": "clause",
                "parent_id": article_id,
                "children_ids": [point_id],
                "article": "Dieu 2",
                "clause": "Khoan 1",
                "path_title": "Dieu 2 > Khoan 1",
                "official_title": "Test law",
                "doc_number": "01/2026/TT-TEST",
            },
        },
        {
            "node_id": point_id,
            "node_type": "point",
            "text": "Diem a la noi dung chi tiet khong duoc tu dong gom.",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": point_id,
                "chunk_id": point_id,
                "node_type": "point",
                "parent_id": clause_id,
                "article": "Dieu 2",
                "clause": "Khoan 1",
                "point": "Diem a",
                "path_title": "Dieu 2 > Khoan 1 > Diem a",
                "official_title": "Test law",
                "doc_number": "01/2026/TT-TEST",
            },
        },
    ]
    return {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}


def _branch_graph():
    article_id = "doc1::article::1"
    clause_id = "doc1::clause::1::2"
    point_a_id = "doc1::point::1::2::a"
    point_b_id = "doc1::point::1::2::b"
    nodes = [
        {
            "node_id": article_id,
            "node_type": "article",
            "text": "Dieu 1. Phe duyet nhiem vu quy hoach.",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": article_id,
                "node_type": "article",
                "children_ids": [clause_id],
                "path_title": "Dieu 1",
                "official_title": "PHE DUYET NHIEM VU QUY HOACH CHUNG THANH PHO HAI PHONG DEN NAM 2050",
                "doc_number": "423/QD-TTG",
                "order_index": 1,
            },
        },
        {
            "node_id": clause_id,
            "node_type": "clause",
            "text": "2. Quan diem, muc tieu quy hoach",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": clause_id,
                "node_type": "clause",
                "parent_id": article_id,
                "children_ids": [point_a_id, point_b_id],
                "path_title": "Dieu 1 > Khoan 2",
                "official_title": "PHE DUYET NHIEM VU QUY HOACH CHUNG THANH PHO HAI PHONG DEN NAM 2050",
                "doc_number": "423/QD-TTG",
                "order_index": 2,
            },
        },
        {
            "node_id": point_a_id,
            "node_type": "point",
            "text": "a) Quan diem:",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": point_a_id,
                "node_type": "point",
                "parent_id": clause_id,
                "children_ids": ["doc1::bullet::a1", "doc1::bullet::a2"],
                "path_title": "Dieu 1 > Khoan 2 > Diem a",
                "doc_number": "423/QD-TTG",
                "order_index": 3,
            },
        },
        {
            "node_id": point_b_id,
            "node_type": "point",
            "text": "b) Muc tieu:",
            "metadata": {
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "node_id": point_b_id,
                "node_type": "point",
                "parent_id": clause_id,
                "children_ids": ["doc1::bullet::b1", "doc1::bullet::b2"],
                "path_title": "Dieu 1 > Khoan 2 > Diem b",
                "doc_number": "423/QD-TTG",
                "order_index": 6,
            },
        },
    ]
    for order, (node_id, parent_id, text) in enumerate(
        [
            ("doc1::bullet::a1", point_a_id, "- Quan diem 1"),
            ("doc1::bullet::a2", point_a_id, "- Quan diem 2"),
            ("doc1::bullet::b1", point_b_id, "- Muc tieu 1"),
            ("doc1::bullet::b2", point_b_id, "- Muc tieu 2"),
        ],
        start=4,
    ):
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "bullet",
                "text": text,
                "metadata": {
                    "artifact_type": "evidence",
                    "doc_id": "doc1",
                    "node_id": node_id,
                    "node_type": "bullet",
                    "parent_id": parent_id,
                    "children_ids": [],
                    "path_title": "Dieu 1 > Khoan 2 > Bullet",
                    "doc_number": "423/QD-TTG",
                    "order_index": order,
                },
            }
        )
    return {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}


def test_direct_clause_fetches_clause_only_without_child_point_text():
    result = retrieve_tree_guided(
        "Khoan 1 Dieu 2 quy dinh gi?",
        _graph(),
        filters={"doc_number": "01/2026/TT-TEST", "article": "Dieu 2", "clause": "Khoan 1"},
        enable_llm=False,
    )

    assert result["mode"] == "tree_guided"
    assert result["route_case"] == "direct_tree_jump"
    assert result["insufficient_context"] is False
    assert len(result["passages"]) == 1
    assert result["passages"][0]["node_id"] == "doc1::clause::2::1"
    assert "noi dung rieng cua khoan" in result["passages"][0]["text"]
    assert "noi dung chi tiet" not in result["passages"][0]["text"]


def test_direct_article_deepens_instead_of_embedding_child_text():
    result = retrieve_tree_guided(
        "Dieu 2 quy dinh gi?",
        _graph(),
        filters={"doc_number": "01/2026/TT-TEST", "article": "Dieu 2"},
        enable_llm=False,
    )

    assert result["route_case"] == "direct_tree_jump"
    assert result["navigation_trace"][0]["decision"] == "deepen_node"
    assert len(result["navigation_trace"][0]["expanded_candidates"]) == 1
    assert any(step["decision"] == "deepen_node" for step in result["navigation_trace"])
    for passage in result["passages"]:
        if passage["node_id"] == "doc1::article::2":
            assert passage["text"] == "Dieu 2. Nguyen tac chung."
        if passage["node_id"] == "doc1::clause::2::1":
            assert passage["text"] == "Khoan 1 quy dinh noi dung rieng cua khoan."


def test_document_title_in_question_infers_doc_number_without_ui_state():
    graph = _graph()
    for node in graph["nodes"]:
        node["metadata"]["official_title"] = (
            "PHE DUYET NHIEM VU QUY HOACH CHUNG THANH PHO HAI PHONG DEN NAM 2050 TAM NHIN DEN NAM 2075"
        )

    match = _infer_document_filter_from_question(
        graph,
        "Quan diem muc tieu quy hoach trong quyet dinh PHE DUYET NHIEM VU QUY HOACH CHUNG THANH PHO HAI PHONG DEN NAM 2050 TAM NHIN DEN NAM 2075",
    )

    assert match["doc_number"] == "01/2026/TT-TEST"
    assert match["matched_by"] == "title_overlap_in_question"


def test_planning_horizon_year_is_not_used_as_document_year_filter():
    question = (
        "Quan \u0111i\u1ec3m, m\u1ee5c ti\u00eau quy ho\u1ea1ch trong quy\u1ebft \u0111\u1ecbnh "
        "PH\u00ca DUY\u1ec6T NHI\u1ec6M V\u1ee4 QUY HO\u1ea0CH CHUNG TH\u00c0NH PH\u1ed0 H\u1ea2I PH\u00d2NG "
        "\u0110\u1ebeN N\u0102M 2050, T\u1ea6M NH\u00ccN \u0110\u1ebeN N\u0102M 2075"
    )

    profile = infer_query_profile(question)

    assert "year" not in profile["filters"]


def test_document_title_match_tolerates_mojibake_question_text():
    title = (
        "PH\u00ca DUY\u1ec6T NHI\u1ec6M V\u1ee4 QUY HO\u1ea0CH CHUNG TH\u00c0NH PH\u1ed0 H\u1ea2I PH\u00d2NG "
        "\u0110\u1ebeN N\u0102M 2050, T\u1ea6M NH\u00ccN \u0110\u1ebeN N\u0102M 2075"
    )
    graph = _branch_graph()
    for node in graph["nodes"]:
        node["metadata"]["official_title"] = title
    question = (
        "Quan \u0111i\u1ec3m, m\u1ee5c ti\u00eau quy ho\u1ea1ch trong quy\u1ebft \u0111\u1ecbnh " + title
    )
    mojibake_question = question.encode("utf-8").decode("latin1")

    match = _infer_document_filter_from_question(graph, mojibake_question)

    assert match["doc_number"] == "423/QD-TTG"
    assert match["matched_by"] == "title_overlap_in_question"


def test_tree_guided_excludes_bundle_nodes_from_candidates_and_evidence():
    section_id = "doc1::roman_section::i"
    bundle_id = "doc1::roman_section::i::section_bundle"
    child_id = "doc1::item::i::1"
    nodes = [
        {
            "node_id": section_id,
            "node_type": "roman_section",
            "text": "I. Tinh hinh benh truyen nhiem nam 2025",
            "metadata": {
                "node_id": section_id,
                "node_type": "roman_section",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "631/QD-BYT",
                "path_title": "PHAN 1 > I. Tinh hinh benh truyen nhiem nam 2025",
                "children_ids": [child_id],
            },
        },
        {
            "node_id": bundle_id,
            "node_type": "section_bundle",
            "text": "I. Tinh hinh benh truyen nhiem nam 2025 1. Tren the gioi ...",
            "metadata": {
                "node_id": bundle_id,
                "node_type": "section_bundle",
                "artifact_type": "section_bundle",
                "doc_id": "doc1",
                "doc_number": "631/QD-BYT",
                "path_title": "PHAN 1 > I. Tinh hinh benh truyen nhiem nam 2025",
                "children_ids": [],
            },
        },
        {
            "node_id": child_id,
            "node_type": "item",
            "text": "1. Tren the gioi",
            "metadata": {
                "node_id": child_id,
                "node_type": "item",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "631/QD-BYT",
                "path_title": "PHAN 1 > I. Tinh hinh benh truyen nhiem nam 2025 > 1",
                "children_ids": [],
            },
        },
    ]
    graph = {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}

    candidates = _iter_summary_candidates(graph)
    result = retrieve_tree_guided("Tinh hinh benh truyen nhiem nam 2025 tren the gioi", graph, enable_llm=False)

    assert bundle_id not in [item["node_id"] for item in candidates]
    assert bundle_id not in [item["node_id"] for item in result["passages"]]


def test_heading_question_prefers_matching_leaf_heading_before_bm25_noise():
    section_id = "doc1::roman_section::i"
    item_id = "doc1::item::i::1"
    other_id = "doc2::item::noise"
    nodes = [
        {
            "node_id": section_id,
            "node_type": "roman_section",
            "text": "I. Tinh hinh benh truyen nhiem nam 2025",
            "metadata": {
                "node_id": section_id,
                "node_type": "roman_section",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "631/QD-BYT",
                "path_title": "PHAN 1 > I. Tinh hinh benh truyen nhiem nam 2025",
                "children_ids": [item_id],
                "level": 2,
            },
        },
        {
            "node_id": item_id,
            "node_type": "item",
            "text": "1. Tren the gioi Nam 2025 tinh hinh benh truyen nhiem dien bien phuc tap.",
            "metadata": {
                "node_id": item_id,
                "node_type": "item",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "631/QD-BYT",
                "path_title": "PHAN 1 > I. Tinh hinh benh truyen nhiem nam 2025 > 1. Tren the gioi",
                "heading_title": "1. Tren the gioi",
                "children_ids": [],
                "level": 3,
            },
        },
        {
            "node_id": other_id,
            "node_type": "item",
            "text": "Benh va nam xuat hien nhieu lan nhung khong dung muc.",
            "metadata": {
                "node_id": other_id,
                "node_type": "item",
                "artifact_type": "evidence",
                "doc_id": "doc2",
                "doc_number": "999/QD-TEST",
                "path_title": "Noisy heading",
                "children_ids": [],
                "level": 1,
            },
        },
    ]
    graph = {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}

    result = retrieve_tree_guided("Tinh hinh benh truyen nhiem nam 2025 tren the gioi", graph, enable_llm=False)

    assert result["route_case"] == "heading_match"
    assert [p["node_id"] for p in result["passages"]] == [item_id]


def test_deepened_branch_fetches_all_direct_leaf_children_not_global_top_k():
    result = retrieve_tree_guided(
        "Quan diem, muc tieu quy hoach trong quyet dinh PHE DUYET NHIEM VU QUY HOACH CHUNG THANH PHO HAI PHONG DEN NAM 2050",
        _branch_graph(),
        enable_llm=False,
    )

    texts = [p["text"] for p in result["passages"]]
    assert texts == ["- Quan diem 1", "- Quan diem 2", "- Muc tieu 1", "- Muc tieu 2"]
    assert len(result["debug_flow"]["relation_evidence_ids"]) == 4
    assert result["debug_flow"]["relation_evidence_action"] == "deepen_node"


def test_relation_fetch_siblings_gets_full_direct_sibling_group_without_bundling():
    parent_id = "doc1::article::1"
    child_ids = [f"doc1::clause::{idx}" for idx in range(1, 5)]
    nodes = [
        {
            "node_id": parent_id,
            "node_type": "article",
            "text": "Dieu 1. Phe duyet De cuong voi cac noi dung chu yeu sau:",
            "metadata": {
                "node_id": parent_id,
                "node_type": "article",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "673/QD-UBND",
                "path_title": "Dieu 1. Phe duyet De cuong voi cac noi dung chu yeu sau:",
                "children_ids": child_ids,
            },
        }
    ]
    for idx, child_id in enumerate(child_ids, start=1):
        nodes.append(
            {
                "node_id": child_id,
                "node_type": "clause",
                "text": f"{idx}. Noi dung {idx}",
                "metadata": {
                    "node_id": child_id,
                    "node_type": "clause",
                    "artifact_type": "evidence",
                    "doc_id": "doc1",
                    "doc_number": "673/QD-UBND",
                    "path_title": f"Dieu 1 > Khoan {idx}",
                    "parent_id": parent_id,
                    "children_ids": [],
                    "order_index": idx,
                },
            }
        )
    graph = {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}

    completed = _full_relation_node_ids(
        graph,
        child_ids[1:],
        "fetch_siblings",
    )

    assert completed == child_ids


def test_deepen_leaf_decision_fetches_direct_siblings(monkeypatch):
    parent_id = "doc1::article::1"
    child_ids = [f"doc1::clause::{idx}" for idx in range(1, 4)]
    nodes = [
        {
            "node_id": parent_id,
            "node_type": "article",
            "text": "Dieu 1. Bai bo toan bo 3 van ban",
            "metadata": {
                "node_id": parent_id,
                "node_type": "article",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "18/2026/QD-UBND",
                "path_title": "Dieu 1. Bai bo toan bo 3 van ban",
                "children_ids": child_ids,
                "order_index": 1,
            },
        }
    ]
    for idx, child_id in enumerate(child_ids, start=1):
        nodes.append(
            {
                "node_id": child_id,
                "node_type": "clause",
                "text": f"{idx}. Van ban {idx}",
                "metadata": {
                    "node_id": child_id,
                    "node_type": "clause",
                    "artifact_type": "evidence",
                    "doc_id": "doc1",
                    "doc_number": "18/2026/QD-UBND",
                    "article": "Dieu 1",
                    "path_title": f"Dieu 1. Bai bo toan bo 3 van ban > Khoan {idx}",
                    "parent_id": parent_id,
                    "children_ids": [],
                    "order_index": idx + 1,
                },
            }
        )
    graph = {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}

    def fake_decision(question, query_profile, candidates, *, hop, enable_llm):
        return {
            "decision": "deepen_node",
            "selected_node_ids": [child_ids[1]],
            "answer_scope": "list",
            "answerability": "partial",
            "needs_reasoning": True,
            "confidence": 0.0,
        }

    monkeypatch.setattr("src.rag.tree_guided_pipeline._llm_navigation_decision", fake_decision)

    result = retrieve_tree_guided("dieu 1 trong van ban lai chau", graph, enable_llm=True, final_top_k=1)

    assert [p["node_id"] for p in result["passages"]] == child_ids
    assert result["debug_flow"]["relation_evidence_action"] == "fetch_siblings"
    assert result["navigation_trace"][0]["decision_adjusted_to"] == "fetch_siblings"


def test_low_confidence_answer_now_on_leaf_fetches_siblings(monkeypatch):
    parent_id = "doc1::article::1"
    child_ids = [f"doc1::clause::{idx}" for idx in range(1, 4)]
    nodes = [
        {
            "node_id": parent_id,
            "node_type": "article",
            "text": "Dieu 1. Bai bo toan bo 3 van ban",
            "metadata": {
                "node_id": parent_id,
                "node_type": "article",
                "artifact_type": "evidence",
                "doc_id": "doc1",
                "doc_number": "18/2026/QD-UBND",
                "path_title": "Dieu 1. Bai bo toan bo 3 van ban",
                "children_ids": child_ids,
                "order_index": 1,
            },
        }
    ]
    for idx, child_id in enumerate(child_ids, start=1):
        nodes.append(
            {
                "node_id": child_id,
                "node_type": "clause",
                "text": f"{idx}. Van ban {idx}",
                "metadata": {
                    "node_id": child_id,
                    "node_type": "clause",
                    "artifact_type": "evidence",
                    "doc_id": "doc1",
                    "doc_number": "18/2026/QD-UBND",
                    "article": "Dieu 1",
                    "path_title": f"Dieu 1. Bai bo toan bo 3 van ban > Khoan {idx}",
                    "parent_id": parent_id,
                    "children_ids": [],
                    "order_index": idx + 1,
                },
            }
        )
    graph = {"nodes": nodes, "edges": [], "node_index": {node["node_id"]: node for node in nodes}}

    def fake_decision(question, query_profile, candidates, *, hop, enable_llm):
        return {
            "decision": "answer_now",
            "selected_node_ids": [child_ids[1]],
            "answer_scope": "list",
            "answerability": "partial",
            "needs_reasoning": True,
            "confidence": 0.2,
        }

    monkeypatch.setattr("src.rag.tree_guided_pipeline._llm_navigation_decision", fake_decision)

    result = retrieve_tree_guided("dieu 1 trong van ban lai chau", graph, enable_llm=True, final_top_k=1)

    assert [p["node_id"] for p in result["passages"]] == child_ids
    assert result["debug_flow"]["relation_evidence_action"] == "fetch_siblings"
    assert result["navigation_trace"][0]["adjustment_reason"] == "low_confidence_graph_relation"


def test_relation_fetch_parent_gets_parent_only():
    graph = _graph()

    assert _full_relation_node_ids(graph, ["doc1::clause::2::1"], "fetch_parent_summary") == ["doc1::article::2"]


def test_answer_source_children_fetches_direct_children(monkeypatch):
    graph = _branch_graph()
    clause_id = "doc1::clause::1::2"

    def fake_decision(question, query_profile, candidates, *, hop, enable_llm):
        return {
            "decision": "answer_now",
            "answer_scope": "list",
            "answer_source": "children",
            "selected_node_ids": [clause_id],
            "answerability": "partial",
            "needs_reasoning": True,
            "confidence": 0.9,
        }

    monkeypatch.setattr("src.rag.tree_guided_pipeline._llm_navigation_decision", fake_decision)

    result = retrieve_tree_guided("quan diem muc tieu", graph, enable_llm=True, final_top_k=4)

    assert result["debug_flow"]["answer_source"] == "children"
    assert result["debug_flow"]["relation_evidence_action"] == "deepen_node"
    assert [p["node_id"] for p in result["passages"]] == ["doc1::point::1::2::a", "doc1::point::1::2::b"]


def test_answer_source_parent_heading_uses_ancestor_heading(monkeypatch):
    graph = _branch_graph()
    child_id = "doc1::point::1::2::a"

    def fake_decision(question, query_profile, candidates, *, hop, enable_llm):
        return {
            "decision": "answer_now",
            "answer_scope": "exact",
            "answer_source": "parent_heading",
            "selected_node_ids": [child_id],
            "answerability": "sufficient",
            "needs_reasoning": False,
            "confidence": 0.9,
        }

    monkeypatch.setattr("src.rag.tree_guided_pipeline._llm_navigation_decision", fake_decision)

    result = retrieve_tree_guided("quyet dinh phe duyet gi", graph, enable_llm=True, final_top_k=1)

    assert result["debug_flow"]["answer_source"] == "parent_heading"
    assert result["passages"][0]["answer_source"] == "parent_heading"
    assert "Phe duyet nhiem vu quy hoach" in result["passages"][0]["text"]
