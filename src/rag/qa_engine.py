from __future__ import annotations

import logging
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
- Nếu có nhiều khả năng (ví dụ hành chính / hình sự / hiệu lực / sửa đổi), hãy tách ý theo từng trường hợp.
- Không được khẳng định quá mức khi dữ liệu chưa đầy đủ.
- Viết bằng tiếng Việt.
- Không tự thêm danh sách nguồn ở cuối, vì hệ thống sẽ trả sources riêng.
""".strip()


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _wants_detailed_answer(question: str, response_mode: str | None = None) -> bool:
    mode = (response_mode or "").strip().lower()
    if mode in {"same_level", "upper_level"}:
        return False
    if mode == "lower_level":
        return True

    q = (question or "").lower()
    detail_keywords = [
        "chi tiết",
        "cụ thể",
        "phân tích",
        "giải thích",
        "ví dụ",
        "đầy đủ",
        "toàn văn",
    ]
    return any(k in q for k in detail_keywords)


def _answer_style_instructions(question: str, response_mode: str | None = None) -> str:
    wants_detail = _wants_detailed_answer(question, response_mode=response_mode)
    mode = (response_mode or "").strip().lower()

    if mode in {"same_level", "upper_level"}:
        return (
            "- Câu hỏi đang ở mức cùng cấp hoặc trên cấp: trả lời ngắn gọn theo các node chính liên quan, sau đó nêu tóm tắt ngắn.\n"
            "- Không đi sâu chi tiết các cấp con nếu người dùng chưa yêu cầu."
        )

    if mode == "lower_level":
        return (
            "- Câu hỏi đang ở mức dưới cấp: trả lời đầy đủ các cấp con liên quan theo đúng thứ tự cha -> con.\n"
            "- Sau phần chi tiết, thêm một đoạn tóm tắt ngắn để người dùng nắm ý chính."
        )

    if wants_detail:
        return (
            "- Ưu tiên trả lời đầy đủ theo cấu trúc pháp lý (Điều -> Khoản -> Điểm nếu có).\n"
            "- Có thể nêu thêm chi tiết quan trọng của từng mục, nhưng vẫn bám sát ngữ cảnh đã truy xuất."
        )

    return (
        "- Trả lời ngắn gọn, bám sát trọng tâm câu hỏi; không lan man.\n"
        "- Nếu câu hỏi yêu cầu 'quy định như thế nào' cho một Điều, hãy gom theo từng mục con (Khoản/Điểm) và tóm tắt 1 ý chính cho mỗi mục.\n"
        "- Chỉ mở rộng giải thích chi tiết khi người dùng yêu cầu rõ."
    )


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
        node_type = md.get("node_type") or p.get("node_type") or p.get("metadata", {}).get("legal_role", "")
        legal_role = md.get("legal_role", "")
        parent_id = md.get("parent_id", "")
        version_status = md.get("version_status", "")

        text = _norm_space(str(p.get("text") or p.get("snippet") or ""))

        lines.append(
            f"[Nguồn {i}] "
            f"node_type={node_type or 'unknown'} | "
            f"legal_role={legal_role or 'unknown'} | "
            f"law_name={law_name or 'unknown'} | "
            f"law_type={law_type or 'unknown'} | "
            f"year={year or 'unknown'} | "
            f"source={source or 'unknown'} | "
            f"article={article or '-'} | "
            f"clause={clause or '-'} | "
            f"point={point or '-'} | "
            f"parent_id={parent_id or '-'} | "
            f"version_status={version_status or '-'}"
        )
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


def _build_user_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    response_mode: str | None = None,
) -> str:
    context = _build_context(passages, settings.answer_max_context_passages)
    answer_style = _answer_style_instructions(question, response_mode=response_mode)

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
{answer_style}
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
    wants_detail = _wants_detailed_answer(question, response_mode=response_mode)
    logger.info(
        "QA decision: response_mode=%s wants_detail=%s passages_in=%s context_limit=%s source_limit=%s",
        response_mode,
        wants_detail,
        len(passages),
        max_context_passages,
        max_source_items,
    )
    preview_nodes: List[str] = []
    for p in passages[:max_context_passages]:
        md = p.get("metadata", {}) or {}
        preview_nodes.append(
            "|".join(
                [
                    str(p.get("node_id") or ""),
                    str(md.get("node_type") or ""),
                    str(md.get("article") or ""),
                    str(md.get("clause") or ""),
                    str(md.get("point") or ""),
                ]
            )
        )
    if preview_nodes:
        logger.info("QA context nodes selected: %s", preview_nodes)

    prompt = _build_user_prompt(
        question,
        passages[:max_context_passages],
        response_mode=response_mode,
    )
    logger.debug("QA prompt preview (first 1200 chars): %s", prompt[:1200])

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
    hierarchy_scope = retrieval_result.get("hierarchy_scope", {}) or {}
    response_mode = hierarchy_scope.get("relative_level")
    return build_chat_response(
        question=question,
        passages=passages,
        max_context_passages=max_context_passages,
        max_source_items=max_source_items,
        response_mode=response_mode,
    )
