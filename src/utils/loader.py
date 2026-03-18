from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text or "" for p in doc.paragraphs)


def read_doc(path: Path) -> str:
    """
    Đọc file .doc (Word 97-2003) qua CLI tool nếu có:
    - antiword (ưu tiên)
    - soffice/libreoffice (fallback)
    """
    antiword = shutil.which("antiword")
    if antiword:
        proc = subprocess.run(
            [antiword, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc.returncode == 0 and (proc.stdout or "").strip():
            return proc.stdout

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            proc = subprocess.run(
                [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", str(outdir), str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc.returncode == 0:
                out_txt = outdir / f"{path.stem}.txt"
                if out_txt.exists():
                    return out_txt.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Cannot read .doc. Install antiword or LibreOffice (soffice).")


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
    if suffix == ".docx":
        return normalize_text(read_docx(path))
    if suffix == ".doc":
        return normalize_text(read_doc(path))
    raise ValueError(f"Unsupported file type: {suffix}")
