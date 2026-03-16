from __future__ import annotations

import re
from pathlib import Path
from typing import List

from pypdf import PdfReader


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)         # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return normalize_text(read_txt(path))
    if suffix == ".pdf":
        return normalize_text(read_pdf_pypdf(path))
    raise ValueError(f"Unsupported file type: {suffix}")
