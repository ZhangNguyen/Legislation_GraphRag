from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from docx import Document
from pypdf import PdfReader

from src.app.settings import settings

router = APIRouter(prefix="/documents", tags=["documents"])

TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml"}
DOCX_SUFFIXES = {".docx"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | DOCX_SUFFIXES | {".pdf"}
MAX_PREVIEW_CHARS = 5000


def _display_name(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.name


def _raw_files() -> List[Path]:
    raw_dir = Path(settings.raw_dir)
    if not raw_dir.exists():
        return []
    return [
        fp
        for fp in raw_dir.rglob("*")
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _read_preview(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:MAX_PREVIEW_CHARS]

    if suffix in DOCX_SUFFIXES:
        doc = Document(str(path))
        content = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        return content[:MAX_PREVIEW_CHARS]

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        chunks: List[str] = []
        total = 0
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            remain = MAX_PREVIEW_CHARS - total
            if remain <= 0:
                break
            snippet = page_text[:remain]
            chunks.append(snippet)
            total += len(snippet)
        return "\n\n".join(chunks)

    return ""


@router.get("")
def list_documents() -> Dict[str, Any]:
    files = sorted(_raw_files())
    raw_dir = Path(settings.raw_dir)
    return {
        "items": [
            {
                "id": fp.stem,
                "filename": fp.name,
                "relative_path": str(fp.relative_to(raw_dir)),
                "title": _display_name(fp),
                "size_bytes": fp.stat().st_size,
                "suffix": fp.suffix.lower(),
            }
            for fp in files
        ],
        "count": len(files),
        "raw_dir": str(raw_dir),
        "exists": raw_dir.exists(),
    }


@router.get("/random")
def random_documents(limit: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    raw_dir = Path(settings.raw_dir)
    files = _raw_files()
    if not files:
        return {"items": [], "count": 0}

    picked = random.sample(files, k=min(limit, len(files)))
    return {
        "items": [
            {
                "filename": fp.name,
                "relative_path": str(fp.relative_to(raw_dir)),
                "title": _display_name(fp),
                "suffix": fp.suffix.lower(),
                "size_bytes": fp.stat().st_size,
            }
            for fp in picked
        ],
        "count": len(picked),
    }


@router.get("/preview")
def preview_document(filename: str = Query(..., min_length=1)) -> Dict[str, Any]:
    raw_dir = Path(settings.raw_dir)
    file_path = (raw_dir / filename).resolve()

    if not raw_dir.exists() or raw_dir.resolve() not in file_path.parents:
        raise HTTPException(status_code=400, detail="Invalid document path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    preview_text = _read_preview(file_path).strip()
    return {
        "filename": file_path.name,
        "relative_path": str(file_path.relative_to(raw_dir.resolve())),
        "title": _display_name(file_path),
        "suffix": file_path.suffix.lower(),
        "preview": preview_text or "(Không trích xuất được nội dung xem nhanh)",
    }
