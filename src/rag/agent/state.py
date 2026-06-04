from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LegalRagState(TypedDict, total=False):
    question: str
    graph: Dict[str, Any]
    filters: Dict[str, Any]
    query_profile: Dict[str, Any]
    retrieval_result: Dict[str, Any]
    passages: List[Dict[str, Any]]
    context_grade: Dict[str, Any]
    answer: str
    sources: List[Dict[str, Any]]
    citation_validation: Dict[str, Any]
    error: Optional[str]
