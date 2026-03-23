
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.openai_clients import get_llm
from src.rag.schemas import ChatResponse, SourceItem

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """
Bạn là trợ lý hỏi đáp pháp luật Việt Nam dựa trên ngữ cảnh đã truy xuất từ hệ thống RAG.

Nguyên tắc:
- Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
- Không bịa thêm thông tin ngoài ngữ cảnh.
- Nếu ngữ cảnh chưa đủ chắc chắn, phải nói rõ là "chưa đủ căn cứ từ dữ liệu đã truy xuất".
- Ưu tiên diễn đạt rõ ràng, ngắn gọn, đúng ngữ nghĩa pháp lý.
- Nếu có nhiều nguồn, phải ưu tiên nguồn chính trước rồi mới dùng nguồn phụ để đối chiếu.
- Không trộn lẫn nội dung giữa các văn bản nếu ngữ cảnh chưa đủ chắc chắn chúng đang nói cùng một vấn đề.
- Viết bằng tiếng Việt.
- Không tự thêm danh sách nguồn ở cuối, vì hệ thống sẽ trả sources riêng.
""".strip()


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _wants_detailed_answer(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ["chi tiết", "cụ thể", "phân tích", "giải thích", "ví dụ", "đầy đủ", "toàn văn"])


def _answer_style_instructions(question: str) -> str:
    if _wants_detailed_answer(question):
        return (
            "- Ưu tiên trả lời đầy đủ theo cấu trúc pháp lý (Điều -> Khoản -> Điểm nếu có).\n"
            "- Khi có nhiều nguồn, nêu rõ nguồn chính và nguồn hỗ trợ."
        )
    return (
        "- Trả lời ngắn gọn, bám sát trọng tâm câu hỏi.\n"
        "- Nếu có nhiều nguồn, ưu tiên chốt câu trả lời theo nguồn chính rồi mới bổ sung nguồn phụ nếu thực sự cần."
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _doc_key_from_md(md: Dict[str, Any]) -> str:
    law_name = str(md.get("law_name") or "").strip().lower()
    if law_name:
        return law_name

    document_id = str(md.get("document_id") or "").strip().lower()
    if document_id:
        return document_id

    file_name = str(md.get("file_name") or md.get("filename") or "").strip().lower()
    if file_name:
        return file_name

    chunk_id = str(md.get("chunk_id") or "").strip().lower()
    if "::" in chunk_id:
        return chunk_id.split("::")[0]

    source_path = str(md.get("source_path") or md.get("file_path") or "").strip().lower()
    if source_path:
        return source_path

    source = str(md.get("source") or "").strip().lower()
    if source:
        return source

    return "unknown"

def _group_passages_for_llm(passages: List[Dict[str, Any]], max_units: int) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for p in passages:
        md = p.get("metadata", {}) or {}
        key = _doc_key_from_md(md)
        bucket = grouped.setdefault(
            key,
            {
                "doc_key": key,
                "doc_title": md.get("law_name") or md.get("source") or "unknown",
                "source": md.get("source") or "unknown",
                "law_type": md.get("law_type") or "unknown",
                "year": md.get("year") or "unknown",
                "items": [],
                "doc_best_score": float(p.get("final_score", p.get("cross_score", 0.0)) or 0.0),
            },
        )
        bucket["doc_best_score"] = max(bucket["doc_best_score"], float(p.get("final_score", p.get("cross_score", 0.0)) or 0.0))
        bucket["items"].append(p)

    docs = list(grouped.values())
    docs.sort(key=lambda x: float(x.get("doc_best_score", 0.0)), reverse=True)

    units: List[Dict[str, Any]] = []
    remaining = max_units
    for doc_idx, doc in enumerate(docs):
        items = sorted(doc["items"], key=lambda x: float(x.get("final_score", x.get("cross_score", 0.0)) or 0.0), reverse=True)
        take_n = 3 if doc_idx == 0 else 2
        picked = items[: min(take_n, remaining)]
        if not picked:
            continue
        units.append({**doc, "items": picked})
        remaining -= len(picked)
        if remaining <= 0:
            break

    return units


def _build_context(passages: List[Dict[str, Any]], max_passages: int) -> str:
    units = _group_passages_for_llm(passages, max_passages)
    lines: List[str] = []

    for doc_idx, unit in enumerate(units, start=1):
        lines.append(
            f"[Nhóm nguồn {doc_idx}] "
            f"law_name={unit['doc_title']} | law_type={unit['law_type']} | year={unit['year']} | source={unit['source']}"
        )

        for i, p in enumerate(unit["items"], start=1):
            md = p.get("metadata", {}) or {}
            node_type = md.get("node_type") or p.get("node_type") or md.get("legal_role") or "unknown"
            article = md.get("article") or "-"
            clause = md.get("clause") or "-"
            point = md.get("point") or "-"
            legal_role = md.get("legal_role") or "unknown"
            version_status = md.get("version_status") or "-"
            text = _norm_space(str(p.get("text") or p.get("snippet") or ""))

            lines.append(
                f"  - [Đoạn {doc_idx}.{i}] node_type={node_type} | legal_role={legal_role} | "
                f"article={article} | clause={clause} | point={point} | version_status={version_status}"
            )
            lines.append(f"    {text}")

        lines.append("")

    return "\n".join(lines).strip()


def _build_user_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    context = _build_context(passages, settings.answer_max_context_passages)
    answer_style = _answer_style_instructions(question)

    return f"""
Câu hỏi người dùng:
{question}

Ngữ cảnh đã truy xuất:
{context}

Yêu cầu trả lời:
- Trả lời trực tiếp câu hỏi.
- Nếu có nhiều nguồn, xác định nguồn chính trước rồi mới dùng nguồn phụ để hỗ trợ/đối chiếu.
- Không trộn lẫn nội dung giữa các văn bản nếu không chắc chắn.
- Nếu ngữ cảnh không đủ, nói rõ giới hạn.
- Không cần liệt kê nguồn ở cuối vì hệ thống sẽ trả sources riêng.
{answer_style}
""".strip()


def _to_source_item(p: Dict[str, Any]) -> SourceItem:
    md = p.get("metadata", {}) or {}
    chunk_id = str(md.get("chunk_id") or md.get("node_id") or p.get("node_id") or "")
    return SourceItem(
        chunk_id=chunk_id,
        law_name=str(md.get("law_name") or ""),
        law_type=str(md.get("law_type") or ""),
        year=_safe_int(md.get("year"), 0),
        source=str(md.get("source") or ""),
        article=md.get("article"),
        snippet=str(p.get("snippet") or p.get("text") or "")[:500],
    )


def build_chat_response(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    max_context_passages: int | None = None,
    max_source_items: int | None = None,
    response_mode: str | None = None,
) -> ChatResponse:
    if not passages:
        return ChatResponse(
            answer="Chưa tìm thấy ngữ cảnh phù hợp để trả lời câu hỏi này trong dữ liệu hiện có.",
            sources=[],
        )

    max_context_passages = max_context_passages or settings.answer_max_context_passages
    max_source_items = max_source_items or settings.answer_max_source_items

    llm = get_llm()
    prompt = _build_user_prompt(question, passages[:max_context_passages])

    response = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    answer = _norm_space(getattr(response, "content", "") or "")
    if not answer:
        answer = "Chưa tạo được câu trả lời từ ngữ cảnh đã truy xuất."

    sources: List[SourceItem] = []
    seen = set()
    for p in passages[:max_source_items]:
        src = _to_source_item(p)
        key = (src.chunk_id, src.source, src.article)
        if key in seen:
            continue
        seen.add(key)
        sources.append(src)

    return ChatResponse(answer=answer, sources=sources)

def answer_with_rag(
    question: str,
    retrieval_result: Dict[str, Any],
    *,
    response_mode: str | None = None,
) -> ChatResponse:
    """
    Backward-compatible entrypoint used by the API layer.

    The retrieval pipeline returns a dict that contains passages and optional
    tuning parameters. This adapter extracts those fields and delegates to
    ``build_chat_response``.
    """
    retrieval_result = retrieval_result or {}
    passages = retrieval_result.get("passages", []) or []

    return build_chat_response(
        question=question,
        passages=passages,
        max_context_passages=retrieval_result.get("final_top_k"),
        max_source_items=retrieval_result.get("final_top_k"),
        response_mode=response_mode or retrieval_result.get("mode"),
    )