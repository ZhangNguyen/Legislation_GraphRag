from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PerfStep:
    name: str
    started_at: float
    ended_at: float | None = None

    @property
    def elapsed_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return round((end - self.started_at) * 1000, 3)


@dataclass
class PerformanceMonitor:
    steps: Dict[str, PerfStep] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self.steps[name] = PerfStep(name=name, started_at=time.perf_counter())

    def stop(self, name: str) -> None:
        step = self.steps.get(name)
        if step is None:
            self.steps[name] = PerfStep(name=name, started_at=time.perf_counter(), ended_at=time.perf_counter())
            return
        step.ended_at = time.perf_counter()

    def report(self) -> Dict[str, Any]:
        return {
            k: {
                "elapsed_ms": v.elapsed_ms,
                "done": v.ended_at is not None,
            }
            for k, v in self.steps.items()
        }
