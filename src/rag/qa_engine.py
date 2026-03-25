from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.openai_clients import get_llm
from src.rag.schemas import ChatResponse, SourceItem

BASE_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời pháp luật Việt Nam.

Nguyên tắc:
- Chỉ dựa trên ngữ cảnh đã truy xuất.
- Trả lời đúng trọng tâm câu hỏi.
- Nếu nguồn chưa đủ, nói rõ chưa đủ căn cứ.
- Không suy diễn vượt quá ngữ cảnh.
- Khi cần, nêu rõ căn cứ thuộc văn bản nào, điều/khoản/điểm nào.
""".strip()

ROUTE_EXTRA_PROMPTS = {
    "heading_list": "- Đây là câu hỏi kiểu liệt kê theo heading/phần/điều. Hãy tổng hợp đầy đủ các ý trong cùng điều, không kéo ý từ điều khác nếu không cùng căn cứ.",
    "version_change": "- Đây là câu hỏi về bãi bỏ/sửa đổi/bổ sung/thay thế/hiệu lực. Hãy xác định đúng văn bản hoặc nội dung bị tác động và nêu căn cứ cụ thể.",
    "direct_reference": "- Đây là câu hỏi tham chiếu trực tiếp. Hãy trả lời đúng ý được hỏi tại điều/khoản/điểm đó và hạn chế lan sang phần khác.",
    "condition_circumstance": "- Đây là câu hỏi về điều kiện/trường hợp. Hãy nêu rõ điều kiện hoặc trường hợp áp dụng, rồi dẫn căn cứ.",
    "factoid": "- Hãy ưu tiên đoạn nào trả lời trực tiếp nhất cho câu hỏi.",
    "document_focus": "- Người dùng đang hỏi xoay quanh một văn bản cụ thể. Hãy ưu tiên ngữ cảnh của đúng văn bản đó.",
}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _doc_key(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("doc_id") or metadata.get("law_name") or metadata.get("official_title") or "unknown").strip().lower()


def _group_for_context(passages: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for p in passages[:max_items]:
        md = dict(p.get("metadata") or {})
        key = _doc_key(md)
        bucket = grouped.setdefault(
            key,
            {
                "doc_key": key,
                "doc_title": md.get("official_title") or md.get("law_name") or "unknown",
                "doc_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "items": [],
                "doc_best_score": float(p.get("final_score", 0.0) or 0.0),
            },
        )
        bucket["doc_best_score"] = max(bucket["doc_best_score"], float(p.get("final_score", 0.0) or 0.0))
        bucket["items"].append(p)

    docs = list(grouped.values())
    docs.sort(key=lambda x: float(x.get("doc_best_score", 0.0)), reverse=True)
    return docs


def _dedup_blocks(texts: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for text in texts:
        block = _norm_space(text)
        if not block:
            continue
        key = block.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _build_context(passages: List[Dict[str, Any]], max_items: int) -> str:
    groups = _group_for_context(passages, max_items)
    lines: List[str] = []
    for idx, group in enumerate(groups, start=1):
        lines.append(f"[Nguồn {idx}] {group['doc_type']} | {group['doc_title']} | source={group['source']}")
        shared_blocks = _dedup_blocks([str(item.get("shared_text") or "") for item in group["items"]])
        if shared_blocks:
            lines.append("[Ngữ cảnh chung của văn bản]")
            for block in shared_blocks:
                lines.append(block)
        for j, item in enumerate(group["items"], start=1):
            md = dict(item.get("metadata") or {})
            lines.append(
                f"  - [Passage {idx}.{j}] artifact={md.get('artifact_type') or '-'} | path={md.get('path_title') or '-'} | article={md.get('article') or '-'} | clause={md.get('clause') or '-'} | point={md.get('point') or '-'}"
            )
            local_text = str(item.get("local_text") or item.get("bundle_text") or item.get("retrieval_text") or item.get("text") or "")
            lines.append(f"    {local_text}")
        lines.append("")
    return "\n".join(lines).strip()


def _to_source_item(p: Dict[str, Any]) -> SourceItem:
    md = dict(p.get("metadata") or {})
    snippet = str(p.get("local_text") or p.get("bundle_text") or p.get("retrieval_text") or p.get("text") or p.get("snippet") or "")[:500]
    return SourceItem(
        chunk_id=str(md.get("chunk_id") or md.get("node_id") or p.get("node_id") or ""),
        law_name=str(md.get("official_title") or md.get("law_name") or ""),
        law_type=str(md.get("doc_type") or md.get("law_type") or "Unknown"),
        year=int(md.get("year") or 0),
        source=str(md.get("source") or md.get("issuing_agency") or ""),
        article=md.get("article"),
        snippet=snippet,
    )


def _same_article_focus(passages: List[Dict[str, Any]]) -> bool:
    articles = {
        _norm_space(str((p.get("metadata") or {}).get("article") or ""))
        for p in passages
        if _norm_space(str((p.get("metadata") or {}).get("article") or ""))
    }
    return len(articles) == 1 and bool(articles)


def build_chat_response(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    max_context_passages: int | None = None,
    max_source_items: int | None = None,
    response_mode: str | None = None,
    query_profile: Dict[str, Any] | None = None,
) -> ChatResponse:
    _ = response_mode
    if not passages:
        return ChatResponse(answer="Chưa tìm thấy ngữ cảnh phù hợp để trả lời câu hỏi này trong dữ liệu hiện có.", sources=[])

    query_profile = query_profile or {}
    route = str(query_profile.get("route") or "factoid")
    max_context_passages = max_context_passages or settings.answer_max_context_passages
    max_source_items = max_source_items or settings.answer_max_source_items
    context = _build_context(passages, max_context_passages)

    route_extra = ROUTE_EXTRA_PROMPTS.get(route, "")
    same_article_hint = ""
    if route == "heading_list" and not _same_article_focus(passages):
        same_article_hint = "- Cảnh báo: ngữ cảnh đang có khả năng lẫn nhiều điều. Nếu chưa đủ chắc chắn cùng một điều, hãy nói rõ chưa đủ căn cứ."

    prompt = f"""
Câu hỏi người dùng:
{question}

Loại truy vấn:
{route}

Ngữ cảnh đã truy xuất:
{context}

Yêu cầu:
- Trả lời trực tiếp và bám sát câu hỏi.
- Ưu tiên nguồn nào sát nhất.
- Nếu thiếu căn cứ thì nói rõ.
{route_extra}
{same_article_hint}
""".strip()

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=BASE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
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


def answer_with_rag(question: str, retrieval_result: Dict[str, Any], *, response_mode: str | None = None) -> ChatResponse:
    retrieval_result = retrieval_result or {}
    passages = retrieval_result.get("passages", []) or []
    return build_chat_response(
        question=question,
        passages=passages,
        max_context_passages=retrieval_result.get("final_top_k"),
        max_source_items=retrieval_result.get("final_top_k"),
        response_mode=response_mode or retrieval_result.get("mode"),
        query_profile=retrieval_result.get("query_profile") or {},
    )
