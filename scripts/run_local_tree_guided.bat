@echo off
echo [%date% %time%] starting run_local_tree_guided.bat >> C:\AI\RAG\outputs\runtime\tree_guided_bat.log
cd /d C:\AI\RAG
echo [%date% %time%] cwd=%cd% >> C:\AI\RAG\outputs\runtime\tree_guided_bat.log
C:\AI\RAG\.venv_codex\Scripts\python.exe C:\AI\RAG\scripts\run_local_tree_guided.py >> C:\AI\RAG\outputs\runtime\tree_guided_bat.log 2>&1
echo [%date% %time%] python exited with %ERRORLEVEL% >> C:\AI\RAG\outputs\runtime\tree_guided_bat.log
