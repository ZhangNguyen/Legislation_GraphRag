@echo off
set QDRANT_URL=http://localhost:6333
set QDRANT_API_KEY=
if "%LOCAL_QDRANT_COLLECTION%"=="" (
  set QDRANT_COLLECTION=legal_chunks
) else (
  set QDRANT_COLLECTION=%LOCAL_QDRANT_COLLECTION%
)
if "%LOCAL_RETRIEVAL_PIPELINE%"=="" (
  set RETRIEVAL_PIPELINE=tree_guided
) else (
  set RETRIEVAL_PIPELINE=%LOCAL_RETRIEVAL_PIPELINE%
)
set PYTHONIOENCODING=utf-8

.\.venv_codex\Scripts\python.exe scripts\run_local_ragas.py %*
