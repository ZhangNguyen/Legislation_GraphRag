"""
Performance Monitoring and Profiling

Tracks and analyzes system performance metrics including:
- Response times
- Memory usage
- Token consumption
- Cache hit rates
- Query performance
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Response time metrics (ms)
    response_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    # Memory metrics
    memory_used_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Throughput metrics
    tokens_used: int = 0
    tokens_per_second: float = 0.0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0

    # Data metrics
    num_documents_retrieved: int = 0
    num_chunks_retrieved: int = 0

    # Quality metrics
    latency_percentile_p95: float = 0.0
    latency_percentile_p99: float = 0.0

    # Operational flags
    error_occurred: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TimingBlock:
    """Context manager for timing code blocks"""
    name: str
    result: Dict[str, float] = field(default_factory=dict)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        self.result[self.name] = elapsed
        return False


class PerformanceMonitor:
    """
    Main performance monitoring class.
    Tracks metrics throughout the request lifecycle.
    """

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.timings: Dict[str, List[float]] = {}
        self.memory_snapshots: List[tuple] = []
        self._start_memory_tracking()

    def _start_memory_tracking(self):
        """Start memory tracking"""
        try:
            tracemalloc.start()
        except:
            pass

    def start_timer(self, name: str) -> float:
        """Start a named timer"""
        if name not in self.timings:
            self.timings[name] = []
        self._timer_start = {name: time.perf_counter()}
        return time.perf_counter()

    def end_timer(self, name: str) -> float:
        """End a named timer and record elapsed time"""
        if name not in self._timer_start:
            return 0.0

        elapsed = (time.perf_counter() - self._timer_start[name]) * 1000  # ms
        self.timings[name].append(elapsed)
        return elapsed

    def record_memory(self):
        """Record current memory usage"""
        try:
            current, peak = tracemalloc.get_traced_memory()
            self.metrics.memory_used_mb = current / 1024 / 1024
            self.metrics.peak_memory_mb = peak / 1024 / 1024
            self.memory_snapshots.append((time.time(), current, peak))
        except:
            pass

    def record_retrieval(
        self,
        num_documents: int,
        num_chunks: int,
        time_ms: float,
    ):
        """Record retrieval metrics"""
        self.metrics.num_documents_retrieved = num_documents
        self.metrics.num_chunks_retrieved = num_chunks
        self.metrics.retrieval_time_ms = time_ms

    def record_generation(
        self,
        tokens_used: int,
        time_ms: float,
    ):
        """Record generation metrics"""
        self.metrics.tokens_used = tokens_used
        self.metrics.generation_time_ms = time_ms

        if time_ms > 0:
            # Rough estimate: assume ~4 chars per token
            self.metrics.tokens_per_second = (tokens_used / (time_ms / 1000)) if time_ms > 0 else 0

    def record_cache(
        self,
        hit_rate: float,
        cache_size_mb: float,
    ):
        """Record cache metrics"""
        self.metrics.cache_hit_rate = hit_rate
        self.metrics.cache_size_mb = cache_size_mb

    def record_error(self, error_message: str):
        """Record an error"""
        self.metrics.error_occurred = True
        self.metrics.error_message = error_message

    def set_response_time(self, time_ms: float):
        """Set total response time"""
        self.metrics.response_time_ms = time_ms

    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named timer"""
        if name not in self.timings or not self.timings[name]:
            return {}

        times = self.timings[name]
        return {
            "count": len(times),
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "sum_ms": sum(times),
        }

    def get_all_timings(self) -> Dict[str, Dict[str, float]]:
        """Get all timing statistics"""
        return {name: self.get_timing_stats(name) for name in self.timings}

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics"""
        self.record_memory()
        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        self.record_memory()
        return {
            "metrics": self.metrics.to_dict(),
            "timings": self.get_all_timings(),
        }


class PerformanceBenchmark:
    """
    Benchmark runner for comparative performance analysis.
    Runs queries multiple times and collects aggregate metrics.
    """

    def __init__(self, name: str = "Benchmark"):
        self.name = name
        self.results: List[PerformanceMetrics] = []
        self.start_time: Optional[float] = None

    def add_result(self, metrics: PerformanceMetrics):
        """Add a result to the benchmark"""
        self.results.append(metrics)

    def run(self, query_func, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Run a query function multiple times and collect metrics.

        Args:
            query_func: Function that returns (result, metrics)
            num_iterations: Number of times to run

        Returns:
            Dictionary of benchmark results
        """
        self.start_time = time.time()

        for _ in range(num_iterations):
            try:
                result, metrics = query_func()
                self.add_result(metrics)
            except Exception as e:
                metrics = PerformanceMetrics()
                metrics.error_occurred = True
                metrics.error_message = str(e)
                self.add_result(metrics)

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        if not self.results:
            return {"error": "No results collected"}

        # Filter out error results for statistics
        valid_results = [r for r in self.results if not r.error_occurred]

        if not valid_results:
            return {"error": "All runs resulted in errors"}

        # Extract timing values
        response_times = [r.response_time_ms for r in valid_results]
        retrieval_times = [r.retrieval_time_ms for r in valid_results]
        generation_times = [r.generation_time_ms for r in valid_results]
        memory_used = [r.memory_used_mb for r in valid_results]
        tokens_used = [r.tokens_used for r in valid_results]

        def get_percentile(data: List[float], percentile: float) -> float:
            """Calculate percentile"""
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int((percentile / 100) * len(sorted_data))
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "benchmark_name": self.name,
            "num_iterations": len(self.results),
            "num_successful": len(valid_results),
            "num_errors": len(self.results) - len(valid_results),
            "response_time_stats": {
                "mean_ms": sum(response_times) / len(response_times),
                "median_ms": sorted(response_times)[len(response_times) // 2],
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "p95_ms": get_percentile(response_times, 95),
                "p99_ms": get_percentile(response_times, 99),
            },
            "retrieval_time_stats": {
                "mean_ms": sum(retrieval_times) / len(retrieval_times),
                "median_ms": sorted(retrieval_times)[len(retrieval_times) // 2],
                "min_ms": min(retrieval_times),
                "max_ms": max(retrieval_times),
            },
            "generation_time_stats": {
                "mean_ms": sum(generation_times) / len(generation_times),
                "median_ms": sorted(generation_times)[len(generation_times) // 2],
                "min_ms": min(generation_times),
                "max_ms": max(generation_times),
            },
            "memory_stats": {
                "mean_mb": sum(memory_used) / len(memory_used),
                "max_mb": max(memory_used),
                "min_mb": min(memory_used),
            },
            "tokens_stats": {
                "total_tokens": sum(tokens_used),
                "mean_tokens": sum(tokens_used) / len(tokens_used),
                "max_tokens": max(tokens_used),
                "min_tokens": min(tokens_used),
            },
        }


class RequestProfiler:
    """
    Profiles individual requests with detailed timing breakdown.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.perf_counter()

    def record_stage(
        self,
        stage_name: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record a processing stage"""
        self.stages[stage_name] = {
            "duration_ms": duration_ms,
            "details": details or {},
            "timestamp": time.perf_counter() - self.start_time,
        }

    def get_report(self) -> Dict[str, Any]:
        """Get profiling report"""
        total_time = sum(s["duration_ms"] for s in self.stages.values())

        breakdown = {}
        for stage_name, stage_data in self.stages.items():
            breakdown[stage_name] = {
                "duration_ms": stage_data["duration_ms"],
                "percentage": (stage_data["duration_ms"] / total_time * 100) if total_time > 0 else 0,
                "details": stage_data["details"],
            }

        return {
            "request_id": self.request_id,
            "total_time_ms": total_time,
            "stages": breakdown,
            "critical_path": self._find_critical_path(),
        }

    def _find_critical_path(self) -> List[str]:
        """Find the critical path (slowest stages)"""
        sorted_stages = sorted(
            self.stages.items(),
            key=lambda x: x[1]["duration_ms"],
            reverse=True
        )
        return [stage[0] for stage in sorted_stages[:3]]
