from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "outputs" / "runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "tree_guided_server.out.log"
    stderr_path = log_dir / "tree_guided_server.err.log"
    pid_path = log_dir / "tree_guided_server.pid"

    python_exe = project_root / ".venv_codex" / "Scripts" / "python.exe"
    runner = project_root / "scripts" / "run_local_tree_guided.py"
    flags = 0
    if sys.platform.startswith("win"):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    stdout = stdout_path.open("ab", buffering=0)
    stderr = stderr_path.open("ab", buffering=0)
    proc = subprocess.Popen(
        [str(python_exe), str(runner)],
        cwd=str(project_root),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        creationflags=flags,
    )
    pid_path.write_text(str(proc.pid), encoding="ascii")
    print(proc.pid)


if __name__ == "__main__":
    main()
