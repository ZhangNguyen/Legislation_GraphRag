from fastapi import FastAPI
from src.app.settings import settings

app = FastAPI(title="Legislation RAG", version="0.1.0")

@app.get("/")
def root():
    return {"message": "Legislation RAG is running. Go to /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "collection": settings.qdrant_collection}
