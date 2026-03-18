"""
Test script for benchmarking and performance monitoring system

This script tests the basic functionality of the benchmarking modules.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from src.eval.benchmark_metrics import (
            RAGEvaluator,
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
        )
        from src.eval.performance_monitor import (
            PerformanceMonitor,
            PerformanceBenchmark,
            RequestProfiler,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_performance_monitor():
    """Test performance monitoring"""
    print("\nTesting PerformanceMonitor...")
    try:
        from src.eval.performance_monitor import PerformanceMonitor
        import time
        
        monitor = PerformanceMonitor()
        
        # Test timer
        monitor.start_timer("test")
        time.sleep(0.01)
        elapsed = monitor.end_timer("test")
        
        assert elapsed > 5, "Timer should record time"
        
        # Test recording metrics
        monitor.set_response_time(100)
        monitor.record_retrieval(num_documents=5, num_chunks=25, time_ms=50)
        monitor.record_generation(tokens_used=100, time_ms=50)
        monitor.record_memory()
        
        metrics = monitor.get_metrics()
        assert metrics.response_time_ms == 100
        assert metrics.num_documents_retrieved == 5
        
        print("✓ PerformanceMonitor works correctly")
        return True
    except Exception as e:
        print(f"✗ PerformanceMonitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """Test benchmark functionality"""
    print("\nTesting PerformanceBenchmark...")
    try:
        from src.eval.performance_monitor import PerformanceBenchmark, PerformanceMonitor
        import time
        
        benchmark = PerformanceBenchmark(name="Test Benchmark")
        
        # Add some results
        for i in range(3):
            monitor = PerformanceMonitor()
            monitor.set_response_time(100 + i * 10)
            benchmark.add_result(monitor.get_metrics())
        
        summary = benchmark.get_summary()
        
        assert summary['num_iterations'] == 3
        assert 'response_time_stats' in summary
        assert summary['response_time_stats']['mean_ms'] > 100
        
        print("✓ PerformanceBenchmark works correctly")
        return True
    except Exception as e:
        print(f"✗ PerformanceBenchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_profiler():
    """Test request profiler"""
    print("\nTesting RequestProfiler...")
    try:
        from src.eval.performance_monitor import RequestProfiler
        
        profiler = RequestProfiler("test_request")
        
        profiler.record_stage("stage1", 50, {"info": "test"})
        profiler.record_stage("stage2", 100, {"info": "test"})
        profiler.record_stage("stage3", 25, {"info": "test"})
        
        report = profiler.get_report()
        
        assert report['total_time_ms'] == 175
        assert len(report['stages']) == 3
        assert report['critical_path'][0] == "stage2"  # slowest
        
        print("✓ RequestProfiler works correctly")
        return True
    except Exception as e:
        print(f"✗ RequestProfiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_metrics():
    """Test benchmark metrics (basic structure)"""
    print("\nTesting Benchmark Metrics...")
    try:
        from src.eval.benchmark_metrics import (
            RAGEvaluator,
            MetricResult,
        )
        
        # Test MetricResult
        result = MetricResult(
            metric_name="test",
            score=0.95,
            reasoning="Test reasoning",
            details={"test": "details"}
        )
        
        assert result.score == 0.95
        assert result.metric_name == "test"
        
        # Test RAGEvaluator can be instantiated
        evaluator = RAGEvaluator()
        assert evaluator.faithfulness is not None
        assert evaluator.answer_relevancy is not None
        assert evaluator.context_precision is not None
        assert evaluator.context_recall is not None
        
        print("✓ Benchmark metrics structures work correctly")
        return True
    except Exception as e:
        print(f"✗ Benchmark metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test that API endpoint modules can be imported"""
    print("\nTesting API endpoints...")
    try:
        from src.app.api.chat import ChatRequestModel
        from src.app.api.benchmark import (
            EvaluationRequest,
            BenchmarkRequest,
            BenchmarkQuery,
        )
        
        # Test request models
        chat_req = ChatRequestModel(
            question="Test?",
            include_metrics=True,
            include_benchmark=True
        )
        
        eval_req = EvaluationRequest(
            question="Test?",
            answer="Test answer",
            contexts=["context"]
        )
        
        assert chat_req.include_metrics == True
        assert eval_req.question == "Test?"
        
        print("✓ API endpoints can be imported and instantiated")
        return True
    except Exception as e:
        print(f"✗ API endpoints test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_examples_module():
    """Test that examples module can be imported"""
    print("\nTesting examples module...")
    try:
        from src.eval import examples
        
        # Should have these functions
        assert hasattr(examples, 'example_evaluation')
        assert hasattr(examples, 'example_performance_monitoring')
        assert hasattr(examples, 'example_benchmarking')
        
        print("✓ Examples module can be imported")
        return True
    except Exception as e:
        print(f"✗ Examples module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("BENCHMARKING & PERFORMANCE MONITORING SYSTEM - TEST SUITE")
    print("="*80)
    
    tests = [
        test_imports,
        test_performance_monitor,
        test_benchmark,
        test_profiler,
        test_benchmark_metrics,
        test_api_endpoints,
        test_examples_module,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Run: python src/eval/examples.py")
        print("2. Start server: uvicorn src.app.main:app --reload")
        print("3. Test endpoints at: http://localhost:8000/docs")
        print("4. Read: BENCHMARKING_GUIDE.md")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
