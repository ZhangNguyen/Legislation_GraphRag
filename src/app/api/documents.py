from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import APIRouter

from src.app.settings import settings

router = APIRouter(prefix="/documents", tags=["documents"])


PDF_SUFFIXES = {".pdf"}


def _display_name(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.name


@router.get("")
def list_documents() -> Dict[str, Any]:
    raw_dir = Path(settings.raw_dir)
    if not raw_dir.exists():
        return {
            "items": [],
            "count": 0,
            "raw_dir": str(raw_dir),
            "exists": False,
        }

    items: List[Dict[str, Any]] = []
    for fp in sorted(raw_dir.glob("*")):
        if not fp.is_file() or fp.suffix.lower() not in PDF_SUFFIXES:
            continue
        stat = fp.stat()
        items.append(
            {
                "id": fp.stem,
                "filename": fp.name,
                "title": _display_name(fp),
                "size_bytes": stat.st_size,
                "view_url": f"/files/raw/{quote(fp.name)}",
            }
        )

    return {
        "items": items,
        "count": len(items),
        "raw_dir": str(raw_dir),
        "exists": True,
    }
