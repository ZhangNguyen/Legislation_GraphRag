from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from src.app.api.router import api_router
from src.app.settings import settings

app = FastAPI(title="Legislation RAG", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

UI_INDEX = Path(__file__).resolve().parent / "statistic" / "index.html"


@app.get("/")
def root():
    if UI_INDEX.exists():
        return FileResponse(UI_INDEX)
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": settings.qdrant_collection,
    }
