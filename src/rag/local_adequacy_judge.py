from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import requests

from src.app.settings import settings

logger = logging.getLogger(__name__)
_OLLAMA_AVAILABLE_CACHE: bool | None = None

GENERIC_INSUFFICIENT_MARKERS = (
    "chưa đủ căn cứ",
    "không đủ căn cứ",
    "không có đủ thông tin",
    "chưa được cung cấp",
    "không thể trả lời",
    "không thể cung cấp",
    "ngữ cảnh hiện tại không cung cấp",
    "ngữ cảnh truy xuất hiện tại chưa đủ",
    "không cung cấp thông tin cụ thể",
    "không cung cấp nội dung chi tiết",
)


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _json_from_text(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _passage_text(passage: Dict[str, Any]) -> str:
    return _norm_space(passage.get("local_text") or passage.get("text") or passage.get("snippet") or "")


def _compact_passages(passages: List[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for passage in passages[:limit]:
        md = dict(passage.get("metadata") or {})
        out.append(
            {
                "path": _norm_space(md.get("path_title") or md.get("title") or md.get("heading_title") or ""),
                "doc_number": _norm_space(md.get("doc_number") or ""),
                "answer_scope": _norm_space(passage.get("answer_scope") or ""),
                "answer_source": _norm_space(passage.get("answer_source") or ""),
                "evidence_role": _norm_space(passage.get("evidence_role") or ""),
                "text": _passage_text(passage)[:900],
            }
        )
    return out


def _question_parts(question: str) -> List[str]:
    q = _norm_space(question.lower())
    parts = re.split(r"\s+và\s+|\s+đồng thời\s+|\s*;\s*", q)
    return [part.strip(" ?.,:;") for part in parts if len(part.strip(" ?.,:;")) >= 12]


def _heuristic_judge(question: str, answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_key = _norm_space(answer).lower()
    evidence_text = " ".join(_passage_text(p) for p in passages).lower()
    if not passages:
        return {
            "adequate": False,
            "issue": "insufficient_evidence",
            "missing_aspects": ["no evidence fetched"],
            "suggested_action": "fetch_more_candidates",
            "reason": "No passages were fetched.",
            "provider": "heuristic",
        }
    if any(marker in answer_key for marker in GENERIC_INSUFFICIENT_MARKERS) and evidence_text:
        return {
            "adequate": False,
            "issue": "off_question",
            "missing_aspects": ["answer refused despite available evidence"],
            "suggested_action": "fetch_more_candidates",
            "reason": "Answer contains a generic insufficient-context refusal while evidence exists.",
            "provider": "heuristic",
        }

    q_parts = _question_parts(question)
    if len(q_parts) >= 2:
        missing = []
        for part in q_parts:
            keywords = [tok for tok in re.findall(r"[\wÀ-ỹ]+", part, flags=re.IGNORECASE) if len(tok) >= 5]
            if keywords and not any(tok in answer_key for tok in keywords[:4]):
                missing.append(part[:90])
        if missing:
            return {
                "adequate": False,
                "issue": "missing_part",
                "missing_aspects": missing[:3],
                "suggested_action": "fetch_more_candidates",
                "reason": "The answer appears to miss part of a multi-part question.",
                "provider": "heuristic",
            }

    return {
        "adequate": True,
        "issue": "",
        "missing_aspects": [],
        "suggested_action": "",
        "reason": "Heuristic found no obvious mismatch.",
        "provider": "heuristic",
    }


def _ollama_available() -> bool:
    global _OLLAMA_AVAILABLE_CACHE
    if _OLLAMA_AVAILABLE_CACHE is not None:
        return _OLLAMA_AVAILABLE_CACHE
    try:
        response = requests.get(
            f"{str(settings.ollama_base_url).rstrip('/')}/api/tags",
            timeout=0.8,
        )
        _OLLAMA_AVAILABLE_CACHE = response.status_code < 500
        return _OLLAMA_AVAILABLE_CACHE
    except Exception:
        _OLLAMA_AVAILABLE_CACHE = False
        return False


def _ollama_judge(question: str, answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
You are a local adequacy judge for a Vietnamese legal RAG system.
Decide whether the answer fully answers the user's question using the provided evidence.
Do not rewrite the answer. Return JSON only.

Return schema:
{{
  "adequate": true,
  "issue": "missing_part|off_question|insufficient_evidence|unsupported_claim|too_broad|",
  "missing_aspects": ["..."],
  "suggested_action": "fetch_more_candidates|fetch_children|fetch_siblings|fetch_parent|answer_retry|",
  "reason": "short reason"
}}

Question: {question}
Answer: {answer}
Evidence: {json.dumps(_compact_passages(passages), ensure_ascii=False)}
""".strip()
    response = requests.post(
        f"{str(settings.ollama_base_url).rstrip('/')}/api/chat",
        json={
            "model": settings.ollama_judge_model,
            "messages": [
                {"role": "system", "content": "Return compact valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 256},
        },
        timeout=float(settings.ollama_judge_timeout_s or 20.0),
    )
    response.raise_for_status()
    payload = response.json()
    content = _norm_space(((payload.get("message") or {}).get("content")) or payload.get("response") or "")
    parsed = _json_from_text(content)
    if not parsed:
        raise RuntimeError("Ollama adequacy judge did not return JSON.")
    parsed["provider"] = "ollama"
    parsed["model"] = settings.ollama_judge_model
    parsed["adequate"] = bool(parsed.get("adequate"))
    if "missing_aspects" not in parsed or not isinstance(parsed.get("missing_aspects"), list):
        parsed["missing_aspects"] = []
    return parsed


def judge_answer_adequacy(question: str, answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not bool(getattr(settings, "enable_local_adequacy_judge", True)):
        return {
            "adequate": True,
            "issue": "",
            "missing_aspects": [],
            "suggested_action": "",
            "reason": "Local adequacy judge disabled.",
            "provider": "disabled",
        }

    provider = str(getattr(settings, "local_judge_provider", "auto") or "auto").lower()
    if provider in {"auto", "ollama"}:
        if provider == "ollama" or _ollama_available():
            try:
                return _ollama_judge(question, answer, passages)
            except Exception as exc:
                logger.info("Local Ollama adequacy judge skipped: %s", exc)
                if provider == "ollama":
                    fallback = _heuristic_judge(question, answer, passages)
                    fallback["ollama_error"] = str(exc)
                    return fallback

    return _heuristic_judge(question, answer, passages)
