from __future__ import annotations

from src.rag.agent.nodes import (
    analyze_question,
    generate_or_refuse,
    grade_context_node,
    retrieve_simple_rrf,
    validate_citation_node,
)
from src.rag.agent.state import LegalRagState


def build_workflow():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(LegalRagState)
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("retrieve_simple_rrf", retrieve_simple_rrf)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("generate_or_refuse", generate_or_refuse)
    graph.add_node("validate_citation", validate_citation_node)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_simple_rrf")
    graph.add_edge("retrieve_simple_rrf", "grade_context")
    graph.add_edge("grade_context", "generate_or_refuse")
    graph.add_edge("generate_or_refuse", "validate_citation")
    graph.add_edge("validate_citation", END)
    return graph.compile()
