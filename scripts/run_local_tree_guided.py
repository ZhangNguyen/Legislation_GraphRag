from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    log_path = project_root / "outputs" / "runtime" / "tree_guided_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["QDRANT_API_KEY"] = ""
    os.environ["QDRANT_COLLECTION"] = os.getenv("LOCAL_QDRANT_COLLECTION", "legal_chunks")
    os.environ["NORMALIZED_DIR"] = os.getenv("LOCAL_NORMALIZED_DIR", "data/raw")
    os.environ["NORMALIZED_GLOB"] = os.getenv("LOCAL_NORMALIZED_GLOB", "*.docx")
    os.environ.setdefault("ENABLE_DENSE_SUMMARY_SEARCH", "0")
    os.environ.setdefault("TREE_BRANCH_LEAF_TOP_K", "12")
    os.environ.setdefault("TREE_MAX_HOPS", "3")
    os.environ["RETRIEVAL_PIPELINE"] = "tree_guided"
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    uvicorn.run(
        "src.app.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
