#!/usr/bin/env python3
"""Clean Generation 2 robustness and reliability tests."""

import sys
import time
import asyncio
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("  ⏳ Circuit breaker pattern...")
    
    try:
        from moe_debugger.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
        
        # Test basic circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker(config)
        
        # Test successful call
        def successful_func():
            return "success"
        
        result = cb.call(successful_func)
        assert result == "success", "Should execute successful function"
        
        # Test failing function
        call_count = 0
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")
        
        # First few calls should fail but execute
        for i in range(2):
            try:
                cb.call(failing_func)
            except ValueError:
                pass  # Expected
        
        # Circuit should now be open
        try:
            cb.call(failing_func)
            assert False, "Should have raised CircuitBreakerOpenError"
        except CircuitBreakerOpenError:
            pass  # Expected
        
        # Test metrics
        metrics = cb.get_metrics()
        assert metrics["state"] == "open", "Circuit should be open"
        assert metrics["failed_calls"] > 0, "Should have failed calls"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_retry_mechanism():
    """Test retry mechanism functionality."""
    print("  ⏳ Retry mechanisms...")
    
    try:
        from moe_debugger.retry_mechanism import RetryMechanism, RetryConfig, RetryExhaustedException
        
        # Test successful retry
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_mechanism = RetryMechanism(config)
        
        attempt_count = 0
        def eventually_successful():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Not ready yet")
            return "success"
        
        result = retry_mechanism.execute(eventually_successful)
        assert result == "success", "Should succeed after retry"
        assert attempt_count == 2, "Should have made 2 attempts"
        
        # Test exhausted retries
        def always_fails():
            raise ValueError("Always fails")
        
        try:
            retry_mechanism.execute(always_fails)
            assert False, "Should have raised RetryExhaustedException"
        except RetryExhaustedException:
            pass  # Expected
        
        # Test retry stats
        stats = retry_mechanism.get_stats()
        assert stats["total_attempts"] > 0, "Should have attempt statistics"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_health_checker():
    """Test health checking system."""
    print("  ⏳ Health checking system...")
    
    try:
        from moe_debugger.health_checker import HealthChecker, HealthStatus
        
        health_checker = HealthChecker(check_interval=1)
        
        # Test custom health check
        def custom_check():
            return HealthStatus.HEALTHY, "All good", {"custom_metric": 100}
        
        health_checker.register_check("custom", custom_check)
        
        # Run specific check
        result = health_checker.run_check("custom")
        assert result.status == HealthStatus.HEALTHY, "Custom check should be healthy"
        assert "All good" in result.message, "Should have correct message"
        
        # Run all checks
        results = health_checker.run_all_checks()
        assert "custom" in results, "Should include custom check"
        assert "cpu" in results, "Should include default CPU check"
        
        # Test overall health
        overall_status = health_checker.get_overall_health()
        assert overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING], "Should have valid overall status"
        
        # Test health report
        report = health_checker.get_health_report()
        assert "overall_status" in report, "Report should include overall status"
        assert "checks" in report, "Report should include individual checks"
        assert "system_metrics" in report, "Report should include system metrics"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_existing_monitoring():
    """Test existing monitoring system integration."""
    print("  ⏳ Existing monitoring integration...")
    
    try:
        from moe_debugger.monitoring import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Test monitor startup
        monitor.start_monitoring()
        assert monitor.is_monitoring, "Monitor should be running"
        
        time.sleep(0.1)  # Brief monitoring period
        
        # Get metrics
        metrics = monitor.get_current_metrics()
        assert "cpu_percent" in metrics, "Should include CPU metrics"
        assert "memory_percent" in metrics, "Should include memory metrics"
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring, "Monitor should be stopped"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_validation_system():
    """Test existing validation system."""
    print("  ⏳ Validation system robustness...")
    
    try:
        from moe_debugger.validation import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test safe string validation
        result = validator.validate_string_input("safe_string", "test", max_length=100)
        assert result == "safe_string", "Should pass safe strings"
        
        # Test dangerous input detection
        try:
            validator.validate_string_input("<script>alert('xss')</script>", "test", max_length=100)
            assert False, "Should have rejected XSS attempt"
        except ValidationError:
            pass  # Expected
        
        # Test numeric validation
        result = validator.validate_numeric_input(42, "test", min_value=0, max_value=100)
        assert result == 42, "Should pass valid numbers"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_logging_robustness():
    """Test logging system robustness."""
    print("  ⏳ Logging system robustness...")
    
    try:
        from moe_debugger.logging_config import setup_logging, get_performance_logger
        import logging
        
        # Setup logging
        setup_logging()
        
        # Test performance logging
        perf_logger = get_performance_logger("test_component")
        
        with perf_logger.timer("test_operation"):
            time.sleep(0.01)
        
        # Test regular logging
        logger = logging.getLogger("test_robustness")
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_integrated_robustness():
    """Test integrated robustness features."""
    print("  ⏳ Integrated robustness...")
    
    try:
        from moe_debugger.circuit_breaker import circuit_breaker, CircuitBreakerConfig
        from moe_debugger.retry_mechanism import retry, RetryConfig
        from moe_debugger.debugger_factory import MoEDebuggerFactory
        
        # Test factory with robustness
        debugger = MoEDebuggerFactory.create_debugger()
        assert debugger is not None, "Should create debugger"
        
        # Test robust function decoration
        @circuit_breaker("test_service", CircuitBreakerConfig(failure_threshold=3))
        @retry(RetryConfig(max_attempts=2, base_delay=0.01))
        def robust_function(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "robust_success"
        
        # Test successful execution
        result = robust_function(should_fail=False)
        assert result == "robust_success", "Should execute successfully"
        
        # Test session robustness
        session = debugger.start_session("robust_test")
        assert session is not None, "Should create session"
        
        # Test metrics collection robustness
        routing_stats = debugger.get_routing_stats()
        assert isinstance(routing_stats, dict), "Should return routing stats"
        
        debugger.end_session()
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def main():
    """Run all Generation 2 robustness tests."""
    
    print("🚀 Running Generation 2 Robustness Tests (Clean)\n")
    print("="*60)
    print("🛡️ Robustness and Reliability Tests")
    print("="*60)
    
    tests = [
        ("Circuit breaker pattern", test_circuit_breaker),
        ("Retry mechanisms", test_retry_mechanism),
        ("Health checking system", test_health_checker),
        ("Existing monitoring integration", test_existing_monitoring),
        ("Validation system robustness", test_validation_system),
        ("Logging system robustness", test_logging_robustness),
        ("Integrated robustness", test_integrated_robustness),
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"  ✅ ({time.time() - start_time:.3f}s)")
                passed += 1
            else:
                print(f"  ❌ ({time.time() - start_time:.3f}s)")
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e} ({time.time() - start_time:.3f}s)")
            failed += 1
    
    total_duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("🎯 GENERATION 2 ROBUSTNESS TEST REPORT")
    print("="*60)
    print(f"📊 Test Statistics:")
    print(f"   • Total Tests: {passed + failed}")
    print(f"   • Passed: {passed} ✅")
    print(f"   • Failed: {failed} ❌")
    print(f"   • Success Rate: {passed/(passed+failed)*100:.1f}%")
    print(f"   • Total Duration: {total_duration:.2f}s")
    
    if failed == 0:
        print("\n✅ ALL GENERATION 2 ROBUSTNESS TESTS PASSED!")
        print("🛡️ System robustness and reliability successfully implemented.")
    else:
        print(f"\n⚠️  {failed} tests failed - system needs attention.")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)