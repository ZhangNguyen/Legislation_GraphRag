from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.citation_validator import validate_citations
from src.rag.openai_clients import get_llm

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _json_loads_from_llm(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _context_from_passages(passages: List[Dict[str, Any]], *, max_chars: int = 8000) -> str:
    blocks: List[str] = []
    total = 0
    for idx, passage in enumerate(passages, start=1):
        md = dict(passage.get("metadata") or {})
        text = _norm_space(passage.get("local_text") or passage.get("text") or passage.get("snippet") or "")
        if not text:
            continue
        path = _norm_space(md.get("path_title") or md.get("title") or md.get("heading_title") or "")
        title = _norm_space(md.get("official_title") or md.get("law_name") or "")
        citation = " ".join(
            x
            for x in [
                _norm_space(md.get("doc_number") or ""),
                _norm_space(md.get("article") or ""),
                _norm_space(md.get("clause") or ""),
                _norm_space(md.get("point") or ""),
            ]
            if x
        )
        block = f"[Source {idx}]\nDocument: {title}\nPath: {path}\nCitation metadata: {citation}\nText: {text}".strip()
        if total + len(block) > max_chars:
            break
        total += len(block)
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def verify_answer(
    *,
    question: str,
    answer: str,
    passages: List[Dict[str, Any]],
    enable_llm: bool = True,
) -> Dict[str, Any]:
    citation_result = validate_citations(answer, passages)
    result: Dict[str, Any] = {
        "valid": bool(citation_result.get("valid")),
        "citation_validation": citation_result,
        "llm_validation": {},
        "reason": "citation_validation",
    }
    if not citation_result.get("valid"):
        return result

    if not enable_llm:
        result["valid"] = True
        return result

    context = _context_from_passages(passages)
    if not context:
        result["valid"] = False
        result["reason"] = "empty_context"
        return result

    prompt = f"""
Verify whether the answer is fully supported by the provided legal context.
Return JSON only:
{{
  "valid": true,
  "unsupported_claims": ["..."],
  "missing_citations": ["..."],
  "reason": "..."
}}

Question: {question}

Answer:
{answer}

Context:
{context}
""".strip()
    try:
        resp = get_llm().invoke(
            [
                SystemMessage(content="You are a strict legal answer verifier. Output valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
        llm_result = _json_loads_from_llm(getattr(resp, "content", "") or "")
    except Exception as exc:
        logger.info("LLM answer verifier skipped: %s", exc)
        llm_result = {}

    if llm_result:
        result["llm_validation"] = llm_result
        result["valid"] = bool(llm_result.get("valid"))
        result["reason"] = str(llm_result.get("reason") or "llm_validation")
    else:
        result["valid"] = True
        result["reason"] = "citation_validation_only"
    return result


def repair_answer(
    *,
    question: str,
    answer: str,
    passages: List[Dict[str, Any]],
    verification: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    context = _context_from_passages(passages)
    if not context:
        repaired = "Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này."
        return repaired, verify_answer(question=question, answer=repaired, passages=passages, enable_llm=False)

    prompt = f"""
Rewrite the answer so every legal claim and citation is supported by the context.
If the context is insufficient, say clearly which part is not supported. Do not invent legal references.

Question: {question}

Draft answer:
{answer}

Verification issues:
{json.dumps(verification, ensure_ascii=False)}

Context:
{context}
""".strip()
    try:
        resp = get_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "You are a conservative Vietnamese legal RAG assistant. "
                        "Use only the provided context and avoid unsupported citations."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        repaired = _norm_space(getattr(resp, "content", "") or "")
    except Exception as exc:
        logger.info("LLM answer repair skipped: %s", exc)
        repaired = ""

    if not repaired:
        repaired = "Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này."

    repaired_verification = verify_answer(
        question=question,
        answer=repaired,
        passages=passages,
        enable_llm=False,
    )
    if not repaired_verification.get("valid"):
        repaired = "Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này."
        repaired_verification = verify_answer(
            question=question,
            answer=repaired,
            passages=passages,
            enable_llm=False,
        )
    return repaired, repaired_verification
