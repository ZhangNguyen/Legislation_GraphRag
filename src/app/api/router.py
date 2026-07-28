from __future__ import annotations

from fastapi import APIRouter

from src.app.api.chat import router as chat_router
from src.app.api.documents import router as documents_router
from src.app.api.ingest import router as ingest_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(documents_router)
api_router.include_router(ingest_router)
