from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["QDRANT_API_KEY"] = ""
    os.environ["QDRANT_COLLECTION"] = os.getenv("LOCAL_QDRANT_COLLECTION", "legal_chunks")
    os.environ["RETRIEVAL_PIPELINE"] = os.getenv("LOCAL_RETRIEVAL_PIPELINE", "tree_guided")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    from src.evaluation.run_all import main as run_all_main

    run_all_main()


if __name__ == "__main__":
    main()
