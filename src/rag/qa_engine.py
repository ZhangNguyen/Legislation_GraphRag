from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.openai_clients import get_llm
from src.rag.schemas import ChatResponse, SourceItem

BASE_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời pháp luật Việt Nam.

Nguyên tắc bắt buộc:
- Chỉ dùng ngữ cảnh đã truy xuất để trả lời.
- Không bịa thêm thông tin không có trong ngữ cảnh.
- Nội dung ngữ cảnh có gì thì trả lời đúng theo đó.
- Không suy diễn vượt quá ngữ cảnh truy xuất.
- Nếu ngữ cảnh chưa đủ để khẳng định thì phải nói rõ là chưa đủ căn cứ.
- Khi có thể, nêu căn cứ theo văn bản, điều, khoản, điểm hoặc vị trí tương ứng.

Quy tắc trình bày:
- Nếu câu hỏi là dạng liệt kê như: nguyên tắc, đối tượng áp dụng, trách nhiệm, bao gồm những gì, gồm những ai, các trường hợp, nội dung, phương thức, hồ sơ, điều kiện, thời hạn, trình tự, thủ tục... thì phải liệt kê đầy đủ tất cả các ý có trong ngữ cảnh.
- Không được chỉ lấy 1 ý tiêu biểu rồi dừng.
- Nếu ngữ cảnh có nhiều ý đánh số hoặc nhiều gạch đầu dòng, hãy giữ cấu trúc liệt kê tương ứng.
- Nếu câu hỏi không phải dạng liệt kê thì trả lời tự nhiên, nhưng vẫn phải bám sát ngữ cảnh truy xuất.
""".strip()

LIST_STYLE_TRIGGERS = [
    "nguyên tắc",
    "đối tượng áp dụng",
    "đối tượng",
    "trách nhiệm",
    "bao gồm",
    "gồm những gì",
    "gồm những ai",
    "gồm ai",
    "các trường hợp",
    "trường hợp",
    "nội dung",
    "phương thức",
    "hồ sơ",
    "điều kiện",
    "thời hạn",
    "trình tự",
    "thủ tục",
]


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _clean_preserve_lines(text: str) -> str:
    if text is None:
        return ""
    lines = []
    for raw in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _restore_inline_numbered_list(text: str) -> str:
    """
    Nếu chunk bị dồn thành một dòng kiểu:
    '1. ... 2. ... 3. ...'
    thì tách lại thành nhiều dòng để LLM nhìn rõ cấu trúc.
    """
    raw = _clean_preserve_lines(text)
    if not raw:
        return ""
    restored = re.sub(r"\s+(\d+\.)\s+", r"\n\1 ", raw)
    restored = re.sub(r"\s+([a-zđ]\))\s+", r"\n\1 ", restored, flags=re.IGNORECASE)
    return restored.strip()


def _postprocess_answer(text: str) -> str:
    raw = _clean_preserve_lines(text)
    if not raw:
        return ""
    raw = _restore_inline_numbered_list(raw)
    return raw.strip()


def _is_list_style_question(question: str) -> bool:
    q = _norm_space(question).lower()
    return any(trigger in q for trigger in LIST_STYLE_TRIGGERS)


def _group_for_context(passages: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for p in passages[:max_items]:
        md = dict(p.get("metadata") or {})
        doc_key = str(p.get("doc_key") or md.get("doc_id") or md.get("official_title") or "unknown")

        bucket = grouped.setdefault(
            doc_key,
            {
                "doc_title": md.get("official_title") or md.get("law_name") or "unknown",
                "doc_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "items": [],
                "doc_best_score": float(p.get("final_score", 0.0) or 0.0),
            },
        )
        bucket["doc_best_score"] = max(
            bucket["doc_best_score"],
            float(p.get("final_score", 0.0) or 0.0),
        )
        bucket["items"].append(p)

    docs = list(grouped.values())
    docs.sort(key=lambda x: float(x.get("doc_best_score", 0.0)), reverse=True)
    return docs


def _build_context(passages: List[Dict[str, Any]], max_items: int) -> str:
    docs = _group_for_context(passages, max_items=max_items)
    blocks: List[str] = []

    for doc in docs:
        parts = [
            f"Văn bản: {doc['doc_title']}",
            f"Loại: {doc['doc_type']}",
            f"Nguồn: {doc['source']}",
        ]

        for idx, item in enumerate(doc["items"], start=1):
            md = dict(item.get("metadata") or {})
            path_title = _norm_space(str(md.get("path_title") or md.get("title") or ""))
            shared_text = _clean_preserve_lines(str(item.get("shared_text") or ""))
            local_text = _restore_inline_numbered_list(
                str(item.get("local_text") or item.get("text") or item.get("snippet") or "")
            )

            section_bits: List[str] = [f"[Đoạn {idx}]"]
            if path_title:
                section_bits.append(f"Vị trí: {path_title}")
            if shared_text:
                section_bits.append(shared_text)
            if local_text:
                section_bits.append(local_text)

            parts.append("\n".join(section_bits))

        blocks.append("\n".join(parts))

    return "\n\n" + ("\n\n---\n\n".join(blocks) if blocks else "")


def _build_sources(passages: List[Dict[str, Any]], max_items: int) -> List[SourceItem]:
    out: List[SourceItem] = []
    seen = set()

    for p in passages[:max_items]:
        md = dict(p.get("metadata") or {})
        chunk_id = str(p.get("node_id") or md.get("chunk_id") or md.get("node_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)

        out.append(
            SourceItem(
                chunk_id=chunk_id,
                law_name=str(md.get("official_title") or md.get("law_name") or "unknown"),
                law_type=str(md.get("doc_type") or md.get("law_type") or "Unknown"),
                year=int(md.get("year") or 0),
                source=str(md.get("source") or md.get("issuing_agency") or "LocalFile"),
                article=str(md.get("article") or "") or None,
                snippet=str(p.get("snippet") or p.get("text") or "")[:700],
            )
        )

    return out


def _build_user_prompt(question: str, context: str) -> str:
    is_list_q = _is_list_style_question(question)

    if is_list_q:
        answer_rules = """
Yêu cầu trả lời:
- Đây là câu hỏi dạng liệt kê.
- Hãy trả lời đúng theo những gì có trong ngữ cảnh truy xuất.
- Phải liệt kê đầy đủ tất cả các ý có trong ngữ cảnh.
- Không được rút còn 1 ý đại diện.
- Nếu ngữ cảnh có các ý đánh số như 1., 2., 3., 4. thì phải nêu đủ các ý đó.
- Nếu ngữ cảnh có nhiều gạch đầu dòng, hãy trình bày lại thành danh sách rõ ràng.
- Sau phần trả lời, có thể nêu căn cứ ngắn gọn theo điều/khoản nếu ngữ cảnh thể hiện rõ.
""".strip()
    else:
        answer_rules = """
Yêu cầu trả lời:
- Hãy trả lời đúng theo ngữ cảnh truy xuất.
- Không cần cố rút gọn.
- Không thêm thông tin ngoài ngữ cảnh.
- Nếu ngữ cảnh chỉ nói được đến đâu thì trả lời đúng đến đó.
- Nếu có thể, nêu căn cứ theo văn bản và vị trí.
""".strip()

    return f"""
Câu hỏi:
{question}

Ngữ cảnh:
{context}

{answer_rules}
""".strip()


def answer_with_rag(question: str, retrieval_result: Dict[str, Any]) -> ChatResponse:
    passages = list(retrieval_result.get("passages") or [])
    insufficient = bool(retrieval_result.get("insufficient_context"))

    if insufficient or not passages:
        return ChatResponse(
            answer="Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này.",
            sources=[],
        )

    context = _build_context(passages, max_items=settings.answer_max_context_passages)
    user_prompt = _build_user_prompt(question, context)

    llm = get_llm()
    resp = llm.invoke(
        [
            SystemMessage(content=BASE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = _postprocess_answer(getattr(resp, "content", "") or "")
    if not answer:
        answer = "Chưa đủ căn cứ để trả lời chính xác từ ngữ cảnh hiện có."

    return ChatResponse(
        answer=answer,
        sources=_build_sources(passages, max_items=settings.answer_max_source_items),
    )