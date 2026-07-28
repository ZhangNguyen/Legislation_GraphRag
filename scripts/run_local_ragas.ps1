$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_API_KEY = ""
if (-not $env:LOCAL_QDRANT_COLLECTION) {
    $env:QDRANT_COLLECTION = "legal_chunks"
} else {
    $env:QDRANT_COLLECTION = $env:LOCAL_QDRANT_COLLECTION
}
if (-not $env:LOCAL_RETRIEVAL_PIPELINE) {
    $env:RETRIEVAL_PIPELINE = "tree_guided"
} else {
    $env:RETRIEVAL_PIPELINE = $env:LOCAL_RETRIEVAL_PIPELINE
}
$env:PYTHONIOENCODING = "utf-8"

.\.venv_codex\Scripts\python.exe scripts\run_local_ragas.py @args
