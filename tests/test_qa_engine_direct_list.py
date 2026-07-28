from src.rag.qa_engine import answer_with_rag


def _passage(idx: int, text: str) -> dict:
    return {
        "node_id": f"doc::clause::{idx}",
        "local_text": text,
        "answer_scope": "list",
        "answer_source": "children",
        "evidence_role": "primary",
        "metadata": {
            "node_id": f"doc::clause::{idx}",
            "chunk_id": f"doc::clause::{idx}",
            "node_type": "clause",
            "doc_number": "18/2026/QD-UBND",
            "official_title": "BAI BO CAC VAN BAN",
            "doc_type": "Quyet dinh",
            "year": 2026,
            "article": "Dieu 1",
            "clause": f"Khoan {idx}",
            "path_title": f"Dieu 1. Bai bo toan bo 3 van ban > Khoan {idx}",
            "order_index": idx,
        },
    }


def test_answer_with_rag_uses_llm_for_list_scope(monkeypatch):
    prompts = []

    class FakeLLM:
        def invoke(self, messages):
            prompts.append(messages[-1].content)

            class Response:
                content = "Điều 1 liệt kê 3 văn bản: 1. Văn bản thứ nhất; 2. Văn bản thứ hai; 3. Văn bản thứ ba."

            return Response()

    monkeypatch.setattr("src.rag.qa_engine.get_llm", lambda: FakeLLM())
    retrieval_result = {
        "passages": [
            _passage(1, "1. Văn bản thứ nhất."),
            _passage(2, "2. Văn bản thứ hai."),
            _passage(3, "3. Văn bản thứ ba."),
        ],
        "debug_flow": {
            "answer_scope": "list",
            "answer_source": "children",
            "relation_evidence_action": "deepen_node",
            "relation_evidence_ids": ["doc::clause::1", "doc::clause::2", "doc::clause::3"],
        },
        "context_grade": {
            "status": "needs_reasoning",
            "navigation_decision": {"answer_scope": "list", "answer_source": "children"},
        },
        "insufficient_context": False,
    }

    response = answer_with_rag("Điều 1 gồm những văn bản nào", retrieval_result)

    assert prompts
    assert "answer_scope:\nlist" in prompts[0]
    assert "Vai trò evidence: primary" in prompts[0]
    assert "Văn bản thứ ba" in prompts[0]
    assert "Điều 1 liệt kê 3 văn bản" in response.answer
    assert len(response.sources) == 3
