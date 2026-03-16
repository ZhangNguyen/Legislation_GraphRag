from pydantic import BaseModel
import os


class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "legal_chunks")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "10"))

    # OCR (deploy-friendly override)
    poppler_path: str = os.getenv("POPPLER_PATH", r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "vie")
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))

    # Rerank (Cross-Encoder local)
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "12"))
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")  # cpu | cuda

settings = Settings()
