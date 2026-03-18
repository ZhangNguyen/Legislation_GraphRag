from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import platform
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

    # Windows fallback: Word COM automation via PowerShell (if MS Word is installed).
    if platform.system().lower().startswith("win"):
        with tempfile.TemporaryDirectory() as td:
            out_txt = Path(td) / f"{path.stem}.txt"
            in_path_ps = str(path).replace("'", "''")
            out_path_ps = str(out_txt).replace("'", "''")
            ps_script = (
                "$word = New-Object -ComObject Word.Application; "
                "$word.Visible = $false; "
                f"$doc = $word.Documents.Open('{in_path_ps}'); "
                f"$doc.SaveAs([ref]'{out_path_ps}', [ref]2); "
                "$doc.Close(); "
                "$word.Quit();"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc.returncode == 0 and out_txt.exists():
                return out_txt.read_text(encoding="utf-8", errors="ignore")

    # Last resort fallback: trích chuỗi printable từ binary để không bỏ sót hoàn toàn.
    raw = path.read_bytes()
    candidates = [
        raw.decode("utf-8", errors="ignore"),
        raw.decode("utf-16le", errors="ignore"),
        raw.decode("latin-1", errors="ignore"),
    ]
    best = max(candidates, key=lambda x: len(re.findall(r"[A-Za-zÀ-ỹ0-9]{2,}", x)))
    best = re.sub(r"[^\x09\x0A\x0D\x20-\x7EÀ-ỹ]+", " ", best)
    best = re.sub(r"\s+", " ", best).strip()
    if best:
        return best

    raise ValueError("Cannot read .doc. Install antiword/LibreOffice/MS Word.")


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
