from __future__ import annotations

from fastapi import APIRouter

from src.app.api.benchmark import router as benchmark_router
from src.app.api.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(benchmark_router)
api_router.include_router(chat_router)
