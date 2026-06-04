from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from src.evaluation.schemas import JUDGE_CACHE, ensure_parent
from src.rag.openai_clients import get_llm


def safe_json_loads(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Judge response must be a JSON object.")
    return data


def load_cache(path: str | Path = JUDGE_CACHE) -> Dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        return {}
    try:
        return dict(json.loads(fp.read_text(encoding="utf-8")))
    except Exception:
        return {}


def save_cache(cache: Dict[str, Any], path: str | Path = JUDGE_CACHE) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key(metric_name: str, sample: Dict[str, Any]) -> str:
    payload = {
        "metric_name": metric_name,
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "contexts": sample.get("contexts", []),
        "ground_truth": sample.get("ground_truth", ""),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def judge_with_llm(metric_name: str, prompt: str, cache: dict | None = None, *, key: Optional[str] = None, use_cache: bool = True) -> dict:
    if use_cache and cache is not None and key and key in cache:
        cached = dict(cache[key])
        parsed = dict(cached.get("parsed_response") or {})
        parsed.setdefault("cached", True)
        return parsed

    try:
        llm = get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = str(getattr(resp, "content", "") or "")
        parsed = safe_json_loads(raw)
        parsed.setdefault("score", None)
        parsed["raw_response"] = raw
        if cache is not None and key:
            cache[key] = {
                "metric_name": metric_name,
                "raw_response": raw,
                "parsed_response": {k: v for k, v in parsed.items() if k != "raw_response"},
            }
        return parsed
    except Exception as exc:
        result = {"score": None, "error": str(exc), "reason": f"Judge failed: {exc}"}
        if cache is not None and key:
            cache[key] = {"metric_name": metric_name, "raw_response": "", "parsed_response": result}
        return result


def _contexts_text(sample: Dict[str, Any]) -> str:
    contexts = list(sample.get("contexts") or [])
    return "\n\n".join(f"[Context {idx}] {ctx}" for idx, ctx in enumerate(contexts, start=1))


def faithfulness_lite(sample: dict, use_cache: bool = True, cache: dict | None = None) -> dict:
    metric = "faithfulness_lite"
    prompt = f"""
Bạn là bộ đánh giá RAG pháp luật Việt Nam.

Nhiệm vụ: chấm Faithfulness.
Chỉ kiểm tra câu trả lời có được hỗ trợ bởi contexts hay không.

Không dùng kiến thức bên ngoài.
Không tự suy luận thêm.
Nếu answer có thông tin không xuất hiện hoặc không suy ra trực tiếp từ contexts, coi là unsupported.

Trả về JSON hợp lệ, không markdown:

{{
  "score": 0.0 đến 1.0,
  "unsupported_claims": ["..."],
  "reason": "..."
}}

Cách chấm:
1.0 = mọi claim quan trọng đều được context hỗ trợ.
0.8 = hầu hết đúng, thiếu chi tiết nhỏ.
0.5 = có nhiều ý chưa được context hỗ trợ.
0.2 = phần lớn câu trả lời không có căn cứ.
0.0 = answer gần như bịa hoặc trái context.

Question:
{sample.get("question", "")}

Answer:
{sample.get("answer", "")}

Contexts:
{_contexts_text(sample)}
""".strip()
    return judge_with_llm(metric, prompt, cache, key=cache_key(metric, sample), use_cache=use_cache)


def answer_relevancy_lite(sample: dict, use_cache: bool = True, cache: dict | None = None) -> dict:
    metric = "answer_relevancy_lite"
    prompt = f"""
Bạn là bộ đánh giá RAG pháp luật Việt Nam.

Nhiệm vụ: chấm Answer Relevancy.
Kiểm tra answer có trả lời đúng trọng tâm question không.

Không cần kiểm tra đúng/sai pháp lý ở metric này.
Chỉ đánh giá mức độ liên quan, trực tiếp và không lan man.

Trả về JSON hợp lệ:

{{
  "score": 0.0 đến 1.0,
  "reason": "..."
}}

Cách chấm:
1.0 = trả lời trực tiếp đúng câu hỏi.
0.8 = trả lời đúng trọng tâm nhưng còn hơi thiếu.
0.5 = có liên quan nhưng lan man hoặc thiếu trọng tâm.
0.2 = chỉ liên quan rất ít.
0.0 = không trả lời câu hỏi.

Question:
{sample.get("question", "")}

Answer:
{sample.get("answer", "")}
""".strip()
    return judge_with_llm(metric, prompt, cache, key=cache_key(metric, sample), use_cache=use_cache)


def answer_correctness_lite(sample: dict, use_cache: bool = True, cache: dict | None = None) -> dict:
    metric = "answer_correctness_lite"
    prompt = f"""
Bạn là bộ đánh giá RAG pháp luật Việt Nam.

Nhiệm vụ: chấm Answer Correctness.
So sánh answer với ground_truth.

Chỉ dùng ground_truth làm đáp án chuẩn.
Không dùng kiến thức bên ngoài.
Không phạt nếu wording khác nhưng cùng nghĩa.
Phạt nặng nếu sai văn bản, sai điều/khoản/điểm, sai đối tượng, sai thời hạn, sai điều kiện, sai trách nhiệm hoặc sai nội dung pháp lý chính.

Trả về JSON hợp lệ:

{{
  "score": 0.0 đến 1.0,
  "missing_points": ["..."],
  "incorrect_points": ["..."],
  "reason": "..."
}}

Cách chấm:
1.0 = đúng đầy đủ.
0.8 = đúng ý chính, thiếu chi tiết nhỏ.
0.5 = đúng một phần nhưng thiếu hoặc sai đáng kể.
0.2 = phần lớn sai.
0.0 = hoàn toàn sai hoặc không trả lời.

Question:
{sample.get("question", "")}

Ground truth:
{sample.get("ground_truth", "")}

Answer:
{sample.get("answer", "")}
""".strip()
    return judge_with_llm(metric, prompt, cache, key=cache_key(metric, sample), use_cache=use_cache)
