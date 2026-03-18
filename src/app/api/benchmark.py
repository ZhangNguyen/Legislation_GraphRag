from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.eval.benchmark_metrics import evaluate_rag_response
from src.eval.performance_monitor import PerformanceMonitor

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class BenchmarkRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: List[str] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    n_generated_questions: int = 3


@router.post("/evaluate")
def benchmark_evaluate(req: BenchmarkRequest) -> Dict[str, Any]:
    perf = PerformanceMonitor()

    perf.start("benchmark_total")
    perf.start("metrics")

    result = evaluate_rag_response(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        reference_answer=req.reference_answer,
        n_generated_questions=req.n_generated_questions,
    )

    perf.stop("metrics")
    perf.stop("benchmark_total")

    result["timings_ms"] = perf.report()
    return result
