from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.app.runtime import rebuild_everything, reindex_qdrant_from_normalized

router = APIRouter(prefix="/ingest", tags=["ingest"])


class ReindexRequest(BaseModel):
    input_dir: Optional[str] = None
    glob_pattern: Optional[str] = None


@router.post("/reindex")
def reindex(req: ReindexRequest) -> Dict[str, Any]:
    return reindex_qdrant_from_normalized(
        input_dir=req.input_dir,
        glob_pattern=req.glob_pattern,
    )


@router.post("/rebuild-all")
def rebuild_all(req: ReindexRequest) -> Dict[str, Any]:
    return rebuild_everything(
        input_dir=req.input_dir,
        glob_pattern=req.glob_pattern,
    )