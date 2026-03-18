from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.openai_clients import get_llm

SUMMARY_SYSTEM_PROMPT = """
Bạn là bộ tóm tắt pháp lý cho GraphRAG từ văn bản pháp luật Việt Nam.

Mục tiêu:
- Tóm tắt ngắn gọn, đúng ngữ nghĩa pháp lý.
- Không bịa thêm thông tin không có trong input.
- Không suy diễn vượt quá nội dung đầu vào.
- Nếu input là nội dung sửa đổi/bổ sung/bãi bỏ/hiệu lực thì phải giữ rõ bản chất đó.
- Ưu tiên giữ các ý quan trọng: chủ thể, hành vi, chế tài, đối tượng áp dụng, hiệu lực, viện dẫn, sửa đổi/bãi bỏ.

Trả về JSON hợp lệ, KHÔNG markdown.

Schema bắt buộc:
{
  "summary": "..."
}

Yêu cầu:
- Viết bằng tiếng Việt.
- Ngắn gọn, rõ nghĩa.
- 1 đến 4 câu tùy lượng thông tin đầu vào.
""".strip()

REL_SUMMARIZES = "SUMMARIZES"
REL_HAS_CHILD_SUMMARY = "HAS_CHILD_SUMMARY"

CHANGE_ROLES = {
    "amendment",
    "repeal",
    "effective",
    "applicability",
    "responsibility",
    "transition",
    "correction",
    "replacement",
}
_SUMMARY_CACHE: Dict[str, str] = {}


def build_hierarchical_summaries(
    graph: Dict[str, Any],
    document_title: Optional[str] = None,
    *,
    include_clause: bool = False,
    include_article: bool = True,
    include_change: bool = True,
    include_community: bool = False,
) -> Dict[str, Any]:
    """
    Fallback no-op builder để giữ compatibility cho runtime.
    Hiện tại trả lại graph gốc; các summary node đã có thể được tạo từ ingestion.
    """
    _ = (
        document_title,
        include_clause,
        include_article,
        include_change,
        include_community,
    )
    return {
        "graph": graph,
        "summary_nodes": [],
        "summary_edges": [],
    }
