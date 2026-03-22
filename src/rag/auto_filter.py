from __future__ import annotations

import re
from typing import Any, Dict

ARTICLE_NUM_RE = re.compile(r"(?:điều)\s*(\d+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
LAW_TYPE_PATTERNS = {
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
    "chỉ thị": "Chỉ thị",
    "công điện": "Công điện",
    "nghị quyết": "Nghị quyết",
}


def infer_filters(question: str) -> Dict[str, Any]:
    """
    Chỉ suy ra các hard filter khi người dùng nêu thật rõ.
    Tránh filter quá tay làm mất recall.
    """
    q = str(question or "")
    q_lower = q.lower()
    out: Dict[str, Any] = {}

    m = ARTICLE_NUM_RE.search(q)
    if m:
        out["article"] = f"Điều {m.group(1)}"

    y = YEAR_RE.search(q)
    if y:
        try:
            out["year"] = int(y.group(1))
        except Exception:
            pass

    for key, value in LAW_TYPE_PATTERNS.items():
        if key in q_lower:
            out["law_type"] = value
            break

    return out
