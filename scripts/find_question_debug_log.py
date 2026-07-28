from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _matches(record: Dict[str, Any], *, request_id: str, query: str) -> bool:
    if request_id and str(record.get("request_id") or "") != request_id:
        return False
    if query:
        q = query.lower()
        question = str((record.get("request") or {}).get("question") or "").lower()
        answer = str((record.get("response") or {}).get("answer") or "").lower()
        if q not in question and q not in answer:
            return False
    return True


def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
    request = record.get("request") or {}
    response = record.get("response") or {}
    return {
        "request_id": record.get("request_id"),
        "timestamp_utc": record.get("timestamp_utc"),
        "stage": record.get("stage"),
        "question": request.get("question"),
        "answer": response.get("answer"),
        "debug_summary": record.get("debug_summary"),
        "timings_ms": record.get("timings_ms"),
        "evidence_hints": record.get("evidence_hints"),
        "error": record.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find per-question debug logs.")
    parser.add_argument("--path", default="outputs/debug/question_debug.jsonl")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)
    records = [
        record
        for record in _load_records(path)
        if _matches(record, request_id=args.request_id.strip(), query=args.query.strip())
    ]
    records = records[-max(1, args.last) :]
    payload = records if args.full else [_compact(record) for record in records]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

