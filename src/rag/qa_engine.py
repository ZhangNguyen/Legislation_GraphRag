from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.openai_clients import get_llm
from src.rag.schemas import ChatResponse, SourceItem


ANSWER_SYSTEM_PROMPT = """
Bạn là trợ lý hỏi đáp pháp luật Việt Nam dựa trên ngữ cảnh đã truy xuất từ hệ thống RAG.

Nguyên tắc:
- Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
- Không bịa thêm thông tin ngoài ngữ cảnh.
- Nếu ngữ cảnh chưa đủ chắc chắn, phải nói rõ là "chưa đủ căn cứ từ dữ liệu đã truy xuất".
- Ưu tiên diễn đạt rõ ràng, ngắn gọn, đúng ngữ nghĩa pháp lý.
- Nếu có nhiều khả năng (ví dụ hành chính / hình sự / hiệu lực / sửa đổi), hãy tách ý theo từng trường hợp.
- Không được khẳng định quá mức khi dữ liệu chưa đầy đủ.
- Viết bằng tiếng Việt.
- Không tự thêm danh sách nguồn ở cuối, vì hệ thống sẽ trả sources riêng.
""".strip()


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _build_context(passages: List[Dict[str, Any]], max_passages: int) -> str:
    lines: List[str] = []

    for i, p in enumerate(passages[:max_passages], start=1):
        md = p.get("metadata", {}) or {}

        law_name = md.get("law_name", "")
        law_type = md.get("law_type", "")
        year = md.get("year", "")
        source = md.get("source", "")
        article = md.get("article", "")
        clause = md.get("clause", "")
        point = md.get("point", "")
        version_status = md.get("version_status", "")

        header_parts = [
            str(x) for x in [
                law_name,
                law_type,
                year,
                source,
                article,
                clause,
                point,
                version_status,
            ]
            if x not in (None, "")
        ]
        header = " | ".join(header_parts) if header_parts else f"Passage {i}"

        text = _norm_space(str(p.get("text") or p.get("snippet") or ""))

        lines.append(f"[Nguồn {i}] {header}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


def _build_user_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    context = _build_context(passages, settings.answer_max_context_passages)

    return f"""
Câu hỏi người dùng:
{question}

Ngữ cảnh đã truy xuất:
{context}

Yêu cầu trả lời:
- Trả lời trực tiếp câu hỏi.
- Nếu có thể, nêu rõ theo từng trường hợp.
- Nếu ngữ cảnh không đủ, nói rõ giới hạn.
- Không cần liệt kê nguồn ở cuối vì nguồn sẽ được hệ thống trả riêng.
""".strip()


def _to_source_item(p: Dict[str, Any]) -> SourceItem:
    md = p.get("metadata", {}) or {}

    chunk_id = str(
        md.get("chunk_id")
        or md.get("node_id")
        or p.get("node_id")
        or ""
    )

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
        sig = (
            src.chunk_id.strip().lower(),
            src.law_name.strip().lower(),
            str(src.article or "").strip().lower(),
        )
        if sig in seen:
            continue
        seen.add(sig)
        sources.append(src)

    return ChatResponse(answer=answer, sources=sources)


def answer_with_rag(
    question: str,
    retrieval_result: Dict[str, Any],
    *,
    max_context_passages: int | None = None,
    max_source_items: int | None = None,
) -> ChatResponse:
    passages = retrieval_result.get("passages", []) or []
    return build_chat_response(
        question=question,
        passages=passages,
        max_context_passages=max_context_passages,
        max_source_items=max_source_items,
    )