from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from src.app.api.router import api_router
from src.app.runtime import preload_runtime_on_startup
from src.app.settings import settings
from src.rag.bm25_service import load_or_build_bm25
import os
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


@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting...")
    bm25_path = settings.bm25_stats_path  # outputs/bm25/bm25_stats.json

    # Nếu chưa có → build
    if not os.path.exists(bm25_path):
        print("⚠️ BM25 stats not found → building...")
        load_or_build_bm25(force_rebuild=True)
        print("✅ BM25 built successfully!")
    else:
        print("✅ BM25 stats loaded")
    preload_runtime_on_startup(
        preload_graph_snapshot=True,
        build_graph_if_missing=True,
        warm_services=True,
    )