from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.openai_clients import get_llm
from src.rag.schemas import ChatResponse, SourceItem

ALLOWED_ANSWER_SCOPES = {"exact", "definition", "summary", "list", "comparison", "procedure"}

BASE_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời pháp luật Việt Nam.

Nguyên tắc bắt buộc:
- Chỉ dùng ngữ cảnh đã truy xuất để trả lời.
- Không bịa thêm thông tin không có trong ngữ cảnh.
- Nội dung ngữ cảnh có gì thì trả lời đúng theo đó.
- Không suy diễn vượt quá ngữ cảnh truy xuất.
- Nếu ngữ cảnh có bằng chứng liên quan trực tiếp, trả lời phần có căn cứ đó; không từ chối toàn bộ chỉ vì bằng chứng có thể chưa bao quát mọi khía cạnh.
- Chỉ nói chưa đủ căn cứ đối với phần thật sự không có trong ngữ cảnh.
- Khi context không rỗng, không được trả lời bằng một câu từ chối chung chung.
- Khi có thể, nêu căn cứ theo văn bản, điều, khoản, điểm hoặc vị trí tương ứng.
""".strip()

LEGAL_REASONING_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời pháp luật Việt Nam.

Nguyên tắc bắt buộc:
- Chỉ sử dụng context đã truy xuất.
- Không bịa số tiền phạt, điều/khoản/điểm, mức trừ điểm, thời hạn hoặc thủ tục nếu context không có.
- Nếu context không đủ, nói rõ phần nào chưa đủ căn cứ.
- Nếu context có bằng chứng liên quan trực tiếp, trả lời phần có căn cứ đó; không từ chối toàn bộ chỉ vì chưa thể khẳng định mọi khía cạnh.
- Khi context không rỗng, không được trả lời bằng một câu từ chối chung chung.
- Được phép suy luận pháp lý thận trọng khi câu hỏi yêu cầu đánh giá tình huống.
- Với câu hỏi ai có lỗi, bên nào sai hoặc ai chịu trách nhiệm trong va chạm: liệt kê hành vi từng bên từ câu hỏi, đối chiếu từng hành vi với quy định được truy xuất, kết luận lỗi có thể là lỗi đơn hoặc lỗi hỗn hợp, và lưu ý tỷ lệ lỗi/trách nhiệm cuối cùng do cơ quan có thẩm quyền xác định.
- Không kết luận chắc chắn nếu context chỉ hỗ trợ một phần.
- Nếu có nhiều khả năng, trình bày theo dạng nếu... thì...
- Khi trả lời, nêu căn cứ theo văn bản/Điều/Khoản/Điểm nếu source có metadata.
- Nếu answer cần citation nhưng sources không có citation rõ, nói "chưa đủ căn cứ trong dữ liệu truy xuất".
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


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _ascii_key(text: Any) -> str:
    normalized = _norm_space(text).lower().replace("đ", "d")
    raw = unicodedata.normalize("NFD", normalized)
    return "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")


def _clean_preserve_lines(text: Any) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _restore_inline_numbered_list(text: Any) -> str:
    raw = _clean_preserve_lines(text)
    if not raw:
        return ""
    restored = re.sub(r"\s+(\d+\.)\s+", r"\n\1 ", raw)
    restored = re.sub(r"\s+([a-zđ]\))\s+", r"\n\1 ", restored, flags=re.IGNORECASE)
    return restored.strip()


def _postprocess_answer(text: str) -> str:
    return _restore_inline_numbered_list(text).strip()


def _navigation_decision(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    context_grade = dict(retrieval_result.get("context_grade") or {})
    decision = dict(context_grade.get("navigation_decision") or {})
    if decision:
        return decision
    debug_flow = dict(retrieval_result.get("debug_flow") or {})
    return {
        "answer_scope": debug_flow.get("answer_scope"),
        "answer_source": debug_flow.get("answer_source"),
        "primary_node_ids": debug_flow.get("primary_node_ids") or [],
        "supporting_node_ids": debug_flow.get("supporting_node_ids") or [],
    }


def _infer_answer_scope(question: str, retrieval_result: Dict[str, Any]) -> str:
    decision = _navigation_decision(retrieval_result)
    scope = str(decision.get("answer_scope") or "").strip().lower()
    if scope in ALLOWED_ANSWER_SCOPES:
        return scope
    debug_scope = str((retrieval_result.get("debug_flow") or {}).get("answer_scope") or "").strip().lower()
    if debug_scope in ALLOWED_ANSWER_SCOPES:
        return debug_scope
    q = _ascii_key(question)
    if any(term in q for term in ("so sanh", "khac nhau", "giong nhau", "doi chieu")):
        return "comparison"
    if any(term in q for term in ("thu tuc", "trinh tu", "ho so", "thoi han", "dieu kien", "quy trinh", "cac buoc")):
        return "procedure"
    if any(
        term in q
        for term in (
            "gom nhung gi",
            "bao gom",
            "liet ke",
            "danh sach",
            "cac noi dung",
            "noi dung chu yeu",
            "cac muc",
            "cac chi tieu",
            "cac nhom",
            "cac loai",
        )
    ):
        return "list"
    if any(term in q for term in ("la gi", "la ai", "duoc hieu la", "dinh nghia", "khai niem")):
        return "definition"
    if _is_doc_scope_summary_question(question):
        return "summary"
    return "exact"


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
    path_title = _norm_space(md.get("path_title") or md.get("title") or "").lower()
    text = _norm_space(item.get("local_text") or item.get("text") or item.get("snippet") or "").lower()
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


def _passage_role(item: Dict[str, Any]) -> str:
    role = str(item.get("evidence_role") or "").strip().lower()
    return "supporting" if role == "supporting" else "primary"


def _select_passages_for_context(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    max_items: int,
    answer_scope: str,
) -> List[Dict[str, Any]]:
    if not passages:
        return []

    if _is_doc_scope_summary_question(question):
        ranked = sorted(passages, key=_score_passage_for_doc_scope, reverse=True)
        return ranked[: min(2, max_items)]

    primary = [p for p in passages if _passage_role(p) == "primary"]
    supporting = [p for p in passages if _passage_role(p) == "supporting"]
    if answer_scope in {"exact", "definition"}:
        selected = primary[: min(len(primary), max_items, 2)]
        if len(selected) < min(max_items, 2) and supporting:
            selected.extend(supporting[: min(len(supporting), min(max_items, 2) - len(selected))])
        return selected or passages[: min(len(passages), max_items, 2)]
    return (primary + supporting)[:max_items]


def _answer_passage_limit(
    question: str,
    passages: List[Dict[str, Any]],
    retrieval_result: Dict[str, Any],
    configured_limit: int,
    answer_scope: str,
) -> int:
    if not passages:
        return 0

    base_limit = max(1, int(configured_limit or 1))
    debug_flow = dict(retrieval_result.get("debug_flow") or {})
    relation_ids = [str(node_id) for node_id in (debug_flow.get("relation_evidence_ids") or []) if str(node_id).strip()]
    relation_action = str(debug_flow.get("relation_evidence_action") or "")

    if answer_scope in {"exact", "definition"}:
        return min(len(passages), max(1, min(base_limit, 2)))
    if answer_scope == "list":
        if relation_ids and relation_action in {"deepen_node", "fetch_siblings"}:
            return min(len(passages), max(base_limit, len(relation_ids), 12))
        return min(len(passages), max(base_limit, 12))
    if answer_scope in {"comparison", "procedure"}:
        return min(len(passages), max(base_limit, 8))
    if _is_doc_scope_summary_question(question):
        return min(len(passages), max(base_limit, 2))
    return min(len(passages), base_limit)


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
        bucket["doc_best_score"] = max(bucket["doc_best_score"], float(p.get("final_score", 0.0) or 0.0))
        bucket["items"].append(p)

    docs = list(grouped.values())
    docs.sort(key=lambda x: float(x.get("doc_best_score", 0.0)), reverse=True)
    return docs


def _build_context(question: str, passages: List[Dict[str, Any]], *, max_items: int, answer_scope: str) -> str:
    selected = _select_passages_for_context(question, passages, max_items=max_items, answer_scope=answer_scope)
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
            path_title = _norm_space(md.get("path_title") or md.get("title") or "")
            shared_text = _clean_preserve_lines(item.get("shared_text") or "")
            local_text = _restore_inline_numbered_list(item.get("local_text") or item.get("text") or item.get("snippet") or "")
            role = _passage_role(item)
            answer_source = _norm_space(item.get("answer_source") or "")

            section_bits: List[str] = [f"[Đoạn {idx}]"]
            section_bits.append(f"Vai trò evidence: {role}")
            section_bits.append(f"answer_scope: {answer_scope}")
            if answer_source:
                section_bits.append(f"Nguồn trả lời ưu tiên: {answer_source}")
            if path_title:
                section_bits.append(f"Vị trí: {path_title}")
            if shared_text:
                section_bits.append(shared_text)
            if local_text:
                section_bits.append(local_text)

            parts.append("\n".join(section_bits))

        blocks.append("\n".join(parts))

    return "\n\n" + ("\n\n---\n\n".join(blocks) if blocks else "")


def _build_sources(question: str, passages: List[Dict[str, Any]], *, max_items: int, answer_scope: str) -> List[SourceItem]:
    selected = _select_passages_for_context(question, passages, max_items=max_items, answer_scope=answer_scope)
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


def _scope_instruction(answer_scope: str) -> str:
    if answer_scope == "exact":
        return "- Đây là câu hỏi exact/factoid. Trả lời thẳng vào ý được hỏi, ngắn gọn; không liệt kê siblings/supporting nếu chúng không phải đáp án trực tiếp."
    if answer_scope == "definition":
        return "- Đây là câu hỏi definition. Nêu định nghĩa/khái niệm đúng theo primary evidence; không mở rộng sang các mục lân cận."
    if answer_scope == "list":
        return "- Đây là câu hỏi list. Liệt kê đầy đủ các primary evidence được cung cấp, giữ thứ tự pháp lý nếu có."
    if answer_scope == "comparison":
        return "- Đây là câu hỏi comparison. So sánh theo từng đối tượng/tiêu chí có trong evidence; không tự thêm tiêu chí ngoài context."
    if answer_scope == "procedure":
        return "- Đây là câu hỏi procedure. Trình bày theo trình tự bước, điều kiện, thời hạn hoặc trách nhiệm có trong evidence."
    return "- Đây là câu hỏi summary. Tóm tắt ngắn gọn ý chính từ primary evidence, chỉ dùng supporting evidence để kiểm tra hoặc bổ sung."


def _build_user_prompt(question: str, context: str, *, context_status: str = "", answer_scope: str = "") -> str:
    scope = answer_scope if answer_scope in ALLOWED_ANSWER_SCOPES else "summary"
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
- Không cần ép buộc theo mẫu liệt kê hay trả lời ngắn nếu answer_scope không yêu cầu.
- Trình bày tự nhiên, miễn là bám sát ngữ cảnh.
- Nếu ngữ cảnh chỉ nói được đến đâu thì trả lời đúng đến đó.
""".strip()

    return f"""
Câu hỏi:
{question}

Trạng thái truy xuất nội bộ:
{context_status or "unknown"}

answer_scope:
{scope}

Ngữ cảnh:
{context}

Quy tắc đánh giá đủ căn cứ:
- Bạn không phải tầng context judge; tầng truy xuất đã chọn evidence ở trên.
- Primary evidence là nguồn chính để trả lời. Supporting evidence chỉ dùng để kiểm tra/bổ sung.
- Không được liệt kê supporting evidence nếu answer_scope không phải list/comparison/procedure.
- Nếu trạng thái truy xuất là sufficient hoặc needs_reasoning, không được trả lời bằng một câu từ chối chung chung.
- Nếu phần Ngữ cảnh có đoạn liên quan trực tiếp đến câu hỏi, hãy trả lời bằng chính nội dung đó.
- Không tự kết luận thiếu căn cứ khi Ngữ cảnh đã có đoạn đánh số, khoản, điểm hoặc câu văn trả lời trực tiếp.
- Nếu đoạn có "Nguồn trả lời ưu tiên: title_or_heading" hoặc "parent_heading", hãy ưu tiên tiêu đề/vị trí pháp lý trong đoạn đó thay vì suy ra từ nội dung node con.
- Với câu hỏi "phê duyệt gì", "ban hành kèm theo gì", "văn bản nào", "đến năm nào", nếu tiêu đề hoặc vị trí Điều 1 đã trả lời trực tiếp thì trả lời ngắn gọn từ tiêu đề/heading đó.
- Nếu chỉ thiếu một phần, trả lời phần có căn cứ và nêu ngắn gọn phần còn thiếu.
{_scope_instruction(scope)}

{extra}

Nếu có thể, nêu căn cứ theo văn bản và vị trí tương ứng.
""".strip()


def answer_with_rag(question: str, retrieval_result: Dict[str, Any]) -> ChatResponse:
    passages = list(retrieval_result.get("passages") or [])
    context_grade = dict(retrieval_result.get("context_grade") or {})

    if not passages:
        return ChatResponse(
            answer="Ngữ cảnh truy xuất hiện tại chưa đủ căn cứ để trả lời chính xác câu hỏi này.",
            sources=[],
        )

    answer_scope = _infer_answer_scope(question, retrieval_result)
    context_limit = _answer_passage_limit(
        question,
        passages,
        retrieval_result,
        settings.answer_max_context_passages,
        answer_scope,
    )
    source_limit = max(int(settings.answer_max_source_items or 1), context_limit)
    raw_context_status = str(context_grade.get("status") or "")
    if raw_context_status == "insufficient" or bool(retrieval_result.get("insufficient_context", False)):
        context_status = "insufficient"
    else:
        context_status = "sufficient"

    context = _build_context(question, passages, max_items=context_limit, answer_scope=answer_scope)
    user_prompt = _build_user_prompt(question, context, context_status=context_status, answer_scope=answer_scope)

    llm = get_llm()
    resp = llm.invoke(
        [
            SystemMessage(
                content=LEGAL_REASONING_SYSTEM_PROMPT
                if str((retrieval_result.get("query_profile") or {}).get("route") or "") == "legal_reasoning"
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
        sources=_build_sources(question, passages, max_items=source_limit, answer_scope=answer_scope),
    )
