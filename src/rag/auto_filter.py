from __future__ import annotations
import re
from typing import Any, Dict

ARTICLE_NUM_RE = re.compile(r"(?:điều)\s*(\d+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Lọc điều, năm từ câu hỏi của người dùng
def infer_filters(question: str) -> Dict[str, Any]:
    filter: Dict[str, Any] = {}

    m = ARTICLE_NUM_RE.search(question)
    if m:
        filter["article"] = f"Điều {m.group(1)}"
    y = YEAR_RE.search(question)
    if y:
        filter["year"] = y.group(1)
    return filter