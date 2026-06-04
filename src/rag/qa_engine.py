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
""".strip()

LEGAL_REASONING_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời pháp luật Việt Nam.

Nguyên tắc bắt buộc:
- Chỉ sử dụng context đã truy xuất.
- Không bịa số tiền phạt, điều/khoản/điểm, mức trừ điểm, thời hạn hoặc thủ tục nếu context không có.
- Nếu context không đủ, nói rõ phần nào chưa đủ căn cứ.
- Được phép suy luận pháp lý thận trọng khi câu hỏi yêu cầu đánh giá tình huống.
- Với câu hỏi ai có lỗi, bên nào sai hoặc ai chịu trách nhiệm trong va chạm: liệt kê hành vi từng bên từ câu hỏi, đối chiếu từng hành vi với quy định được truy xuất, kết luận lỗi có thể là lỗi đơn hoặc lỗi hỗn hợp, và lưu ý tỷ lệ lỗi/trách nhiệm cuối cùng do cơ quan có thẩm quyền xác định.
- Không kết luận chắc chắn nếu context chỉ hỗ trợ một phần.
- Nếu có nhiều khả năng, trình bày theo dạng nếu... thì...
- Khi trả lời, nêu căn cứ theo văn bản/Điều/Khoản/Điểm nếu source có metadata.
- Nếu answer cần citation nhưng sources không có citation rõ, nói “chưa đủ căn cứ trong dữ liệu truy xuất”.
- Không dùng kiến thức ngoài context để thêm mức phạt hoặc điều luật.
- Không tự tạo văn bản pháp luật không có trong sources.
- Không tự suy ra số điều/khoản/điểm.
- Không trả lời lan man ngoài câu hỏi.
- Ưu tiên trả lời ngắn gọn, trực tiếp, có căn cứ.
""".strip()

DOC_SCOPE_SUMMARY_TRIGGERS = [
    "quy định về vấn đề gì",
    "quy định về gì",
    "nói về vấn đề gì",
    "nói về gì",
    "điều chỉnh vấn đề gì",
    "điều chỉnh gì",
    "về vấn đề gì",
    "phạm vi điều chỉnh là gì",
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


def _is_doc_scope_summary_question(question: str) -> bool:
    q = _norm_space(question).lower()
    if not q:
        return False
    if not any(t in q for t in DOC_SCOPE_SUMMARY_TRIGGERS):
        return False
    return any(x in q for x in ["văn bản", "thông tư", "nghị định", "quyết định", "luật", "bộ luật", "nghị quyết"])


def _score_passage_for_doc_scope(item: Dict[str, Any]) -> tuple:
    md = dict(item.get("metadata") or {})
    artifact = str(md.get("artifact_type") or "").lower()
    path_title = _norm_space(str(md.get("path_title") or md.get("title") or "")).lower()
    text = _norm_space(str(item.get("local_text") or item.get("text") or item.get("snippet") or "")).lower()
    final_score = float(item.get("final_score", 0.0) or 0.0)

    artifact_rank = 0
    if artifact == "doc_sketch":
        artifact_rank = 4
    elif artifact == "article_bundle" and ("điều 1" in path_title or "phạm vi điều chỉnh" in text):
        artifact_rank = 3
    elif artifact == "evidence" and ("điều 1" in path_title or "phạm vi điều chỉnh" in text):
        artifact_rank = 2
    elif "phạm vi điều chỉnh" in text:
        artifact_rank = 1

    path_rank = 0
    if "điều 1" in path_title:
        path_rank += 2
    if "phạm vi điều chỉnh" in path_title:
        path_rank += 2

    text_rank = 0
    if "phạm vi điều chỉnh" in text:
        text_rank += 3
    if text.startswith("điều 1."):
        text_rank += 1

    return (artifact_rank, path_rank, text_rank, final_score)


def _select_passages_for_context(question: str, passages: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    if not passages:
        return []

    if _is_doc_scope_summary_question(question):
        ranked = sorted(passages, key=_score_passage_for_doc_scope, reverse=True)
        return ranked[: min(2, max_items)]

    return passages[:max_items]


def _group_for_context(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for p in passages:
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


def _build_context(question: str, passages: List[Dict[str, Any]], max_items: int) -> str:
    selected = _select_passages_for_context(question, passages, max_items=max_items)
    docs = _group_for_context(selected)
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


def _build_sources(question: str, passages: List[Dict[str, Any]], max_items: int) -> List[SourceItem]:
    selected = _select_passages_for_context(question, passages, max_items=max_items)
    out: List[SourceItem] = []
    seen = set()

    for p in selected:
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
    if _is_doc_scope_summary_question(question):
        extra = """
Yêu cầu bổ sung cho loại câu hỏi này:
- Đây là câu hỏi hỏi chủ đề hoặc phạm vi chính của văn bản.
- Ưu tiên trả lời gọn trong 1 câu nếu ngữ cảnh đã đủ rõ.
- Chỉ nêu nội dung chính của văn bản.
- Không mở rộng sang các điều khoản chi tiết khác nếu câu hỏi không yêu cầu.
""".strip()
    else:
        extra = """
Yêu cầu bổ sung:
- Trả lời dựa hoàn toàn trên ngữ cảnh truy xuất.
- Không thêm thông tin ngoài ngữ cảnh.
- Không cần cố rút gọn.
- Không cần ép buộc theo mẫu liệt kê hay trả lời ngắn.
- Trình bày tự nhiên, miễn là bám sát ngữ cảnh.
- Nếu ngữ cảnh chỉ nói được đến đâu thì trả lời đúng đến đó.
""".strip()

    return f"""
Câu hỏi:
{question}

Ngữ cảnh:
{context}

{extra}

Nếu có thể, nêu căn cứ theo văn bản và vị trí tương ứng.
""".strip()


def answer_with_rag(question: str, retrieval_result: Dict[str, Any]) -> ChatResponse:
    passages = list(retrieval_result.get("passages") or [])
    insufficient = bool(retrieval_result.get("insufficient_context"))
    context_grade = dict(retrieval_result.get("context_grade") or {})

    if insufficient or context_grade.get("status") == "insufficient" or not passages:
        return ChatResponse(
            answer="Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này.",
            sources=[],
        )

    context = _build_context(question, passages, max_items=settings.answer_max_context_passages)
    user_prompt = _build_user_prompt(question, context)

    llm = get_llm()
    resp = llm.invoke(
        [
            SystemMessage(
                content=LEGAL_REASONING_SYSTEM_PROMPT
                if context_grade.get("status") == "needs_reasoning"
                else BASE_SYSTEM_PROMPT
            ),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = _postprocess_answer(getattr(resp, "content", "") or "")
    if not answer:
        answer = "Chưa đủ căn cứ để trả lời chính xác từ ngữ cảnh hiện có."

    return ChatResponse(
        answer=answer,
        sources=_build_sources(question, passages, max_items=settings.answer_max_source_items),
    )
