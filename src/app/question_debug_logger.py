from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.app.settings import settings

logger = logging.getLogger(__name__)


def new_question_log_id() -> str:
    return uuid.uuid4().hex


def _jsonable(value: Any, *, depth: int = 0, max_depth: int = 12) -> Any:
    if depth > max_depth:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(), depth=depth + 1, max_depth=max_depth)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth=depth + 1, max_depth=max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth=depth + 1, max_depth=max_depth) for v in value]
    return repr(value)


def _short_text(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _passage_hints(passages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    for idx, passage in enumerate(passages, start=1):
        md = dict(passage.get("metadata") or {})
        hints.append(
            {
                "rank": idx,
                "node_id": passage.get("node_id") or passage.get("id") or md.get("node_id") or md.get("chunk_id"),
                "chunk_id": passage.get("chunk_id") or md.get("chunk_id"),
                "doc_id": passage.get("doc_id") or md.get("doc_id"),
                "doc_number": md.get("doc_number"),
                "official_title": md.get("official_title") or md.get("law_name"),
                "doc_type": md.get("doc_type") or md.get("law_type"),
                "year": md.get("year"),
                "source": md.get("source") or md.get("issuing_agency"),
                "filename": md.get("filename") or md.get("file_name") or md.get("source_file"),
                "path_title": md.get("path_title") or md.get("title") or md.get("heading_title"),
                "article": md.get("article"),
                "clause": md.get("clause"),
                "point": md.get("point"),
                "node_type": md.get("node_type"),
                "artifact_type": md.get("artifact_type"),
                "text_length": len(str(passage.get("local_text") or passage.get("text") or passage.get("snippet") or "")),
                "snippet": _short_text(passage.get("local_text") or passage.get("text") or passage.get("snippet") or ""),
            }
        )
    return hints


def _debug_summary(retrieval_result: Optional[Dict[str, Any]], response_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval_result = retrieval_result or {}
    response_payload = response_payload or {}
    debug_flow = dict(retrieval_result.get("debug_flow") or {})
    context_grade = dict(retrieval_result.get("context_grade") or {})
    nav_decision = dict(context_grade.get("navigation_decision") or {})
    return {
        "pipeline": retrieval_result.get("mode") or retrieval_result.get("route_case"),
        "route_case": retrieval_result.get("route_case"),
        "answer_preview": _short_text(response_payload.get("answer"), 700),
        "sources_count": len(response_payload.get("sources") or []),
        "passages_count": len(retrieval_result.get("passages") or []),
        "insufficient_context": retrieval_result.get("insufficient_context"),
        "context_status": context_grade.get("status"),
        "context_reason": context_grade.get("reason"),
        "navigation_decision": nav_decision.get("decision"),
        "navigation_answerability": nav_decision.get("answerability"),
        "navigation_confidence": nav_decision.get("confidence"),
        "relation_evidence_action": debug_flow.get("relation_evidence_action"),
        "relation_evidence_count": len(debug_flow.get("relation_evidence_ids") or []),
        "post_answer_valid": ((response_payload.get("debug") or {}).get("post_answer_verification") or {}).get("valid"),
    }


def write_question_debug_log(
    *,
    request_id: str,
    request: Dict[str, Any],
    stage: str,
    timings_ms: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    response_payload: Optional[Dict[str, Any]] = None,
    verification: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    try:
        passages = list((retrieval_result or {}).get("passages") or [])
        record = {
            "schema_version": 1,
            "request_id": request_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "request": _jsonable(request),
            "debug_summary": _debug_summary(retrieval_result, response_payload),
            "timings_ms": _jsonable(timings_ms or {}),
            "response": _jsonable(response_payload or {}),
            "verification": _jsonable(verification or {}),
            "retrieval_result": _jsonable(retrieval_result or {}),
            "evidence_hints": _passage_hints(passages),
            "error": None,
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }

        path = Path(settings.question_debug_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.warning("Failed to write question debug log: %s", exc)

