from __future__ import annotations

from typing import Any, Dict

from src.rag.citation_validator import validate_citations
from src.rag.context_grader import grade_context
from src.rag.qa_engine import answer_with_rag
from src.rag.retrieval_pipeline_simple import analyze_query
from src.rag.tree_guided_pipeline import retrieve_tree_guided


def analyze_question(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"query_profile": analyze_query(str(state.get("question") or ""))}


def retrieve_simple_rrf(state: Dict[str, Any]) -> Dict[str, Any]:
    result = retrieve_tree_guided(
        question=str(state.get("question") or ""),
        graph=dict(state.get("graph") or {}),
        filters=dict(state.get("filters") or {}),
    )
    return {
        "retrieval_result": result,
        "passages": list(result.get("passages") or []),
        "query_profile": dict(result.get("query_profile") or state.get("query_profile") or {}),
    }


def grade_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    context_grade = grade_context(
        question=str(state.get("question") or ""),
        query_profile=dict(state.get("query_profile") or {}),
        passages=list(state.get("passages") or []),
    )
    retrieval_result = dict(state.get("retrieval_result") or {})
    retrieval_result["context_grade"] = context_grade
    retrieval_result["insufficient_context"] = context_grade.get("status") == "insufficient"
    return {"context_grade": context_grade, "retrieval_result": retrieval_result}


def generate_or_refuse(state: Dict[str, Any]) -> Dict[str, Any]:
    context_grade = dict(state.get("context_grade") or {})
    if context_grade.get("status") == "insufficient":
        return {
            "answer": "Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này.",
            "sources": [],
        }
    response = answer_with_rag(
        question=str(state.get("question") or ""),
        retrieval_result=dict(state.get("retrieval_result") or {}),
    )
    return {
        "answer": response.answer,
        "sources": [source.model_dump() for source in response.sources],
    }


def validate_citation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sources = list(state.get("passages") or state.get("sources") or [])
    validation = validate_citations(str(state.get("answer") or ""), sources)
    return {"citation_validation": validation}
