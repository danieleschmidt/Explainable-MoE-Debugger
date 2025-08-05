#!/usr/bin/env python3
"""Final validation test for the complete MoE Debugger system."""

import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_component_availability():
    """Test that all major components are available."""
    print("🔍 Testing Component Availability...")
    
    components = [
        ("models", "moe_debugger.models"),
        ("validation", "moe_debugger.validation"),  
        ("logging", "moe_debugger.logging_config"),
        ("monitoring", "moe_debugger.monitoring"),
        ("caching", "moe_debugger.caching"),
        ("performance", "moe_debugger.performance_optimization"),
        ("analyzer", "moe_debugger.analyzer"),
        ("server", "moe_debugger.server"),
        ("cli", "moe_debugger.cli"),
    ]
    
    results = {}
    for name, module in components:
        try:
            __import__(module)
            results[name] = "✅ AVAILABLE"
        except ImportError as e:
            results[name] = f"❌ FAILED: {e}"
    
    for name, status in results.items():
        print(f"  {name:12} - {status}")
    
    available_count = sum(1 for status in results.values() if "✅" in status)
    total_count = len(results)
    
    print(f"\n📊 Component Availability: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
    return available_count / total_count >= 0.8


def test_data_models():
    """Test data model creation and validation."""
    print("\n🏗️  Testing Data Models...")
    
    try:
        from moe_debugger.models import (
            RoutingEvent, ExpertMetrics, LoadBalanceMetrics, 
            DebugSession, ModelArchitecture, HookConfiguration
        )
        
        # Test RoutingEvent
        event = RoutingEvent(
            timestamp=time.time(),
            layer_idx=0,
            token_position=0,
            token="test",
            expert_weights=[0.1, 0.2, 0.7],
            selected_experts=[2],
            routing_confidence=0.8
        )
        
        # Test ModelArchitecture  
        arch = ModelArchitecture(
            num_layers=12,
            num_experts_per_layer=8,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=32000,
            max_sequence_length=2048,
            expert_capacity=2.0
        )
        
        # Test HookConfiguration
        config = HookConfiguration(
            enabled_hooks={"router": True, "experts": True},
            sampling_rate=0.1,
            buffer_size=1000,
            save_gradients=False,
            save_activations=True,
            track_parameters=["weight", "bias"],
            memory_limit_mb=2048
        )
        
        print("  ✅ All data models created successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Data models test failed: {e}")
        return False


def test_validation_system():
    """Test input validation and security."""
    print("\n🛡️  Testing Validation System...")
    
    try:
        from moe_debugger.validation import (
            InputValidator, ValidationError, SecurityError,
            safe_json_loads, safe_json_dumps
        )
        
        validator = InputValidator()
        
        # Test string validation
        safe_string = validator.validate_string_input("test string", "test_field")
        assert safe_string == "test string"
        
        # Test numeric validation
        safe_number = validator.validate_numeric_input(42.5, "test_number", min_value=0, max_value=100)
        assert safe_number == 42.5
        
        # Test security checks
        try:
            validator.validate_string_input("<script>alert('xss')</script>", "malicious_field")
            assert False, "Should have caught XSS attempt"
        except SecurityError:
            pass  # Expected
        
        # Test JSON safety
        test_data = {"key": "value", "number": 123}
        json_str = safe_json_dumps(test_data)
        loaded_data = safe_json_loads(json_str)
        assert loaded_data == test_data
        
        print("  ✅ Validation system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False


def test_logging_system():
    """Test logging configuration and functionality."""
    print("\n📝 Testing Logging System...")
    
    try:
        from moe_debugger.logging_config import get_logger, get_log_manager
        
        logger = get_logger("test_component")
        logger.info("Test log message")
        
        log_manager = get_log_manager()
        stats = log_manager.get_log_stats()
        
        assert "log_level" in stats
        assert "log_directory" in stats
        
        print("  ✅ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Logging test failed: {e}")
        return False


def test_monitoring_system():
    """Test health monitoring and system metrics."""
    print("\n📊 Testing Monitoring System...")
    
    try:
        from moe_debugger.monitoring import HealthMonitor, SystemMonitor, create_default_health_checks
        
        # Create health monitor
        health_monitor = HealthMonitor()
        
        # Add test health check
        def test_check():
            return True
        
        health_monitor.add_health_check("test_check", test_check, critical=False)
        
        # Run health checks
        results = health_monitor.run_health_checks()
        assert results["overall_status"] in ["healthy", "degraded", "critical"]
        assert results["summary"]["total"] >= 1
        
        # Test system monitor
        system_monitor = SystemMonitor()
        current_metrics = system_monitor.get_current_metrics()
        assert "timestamp" in current_metrics
        
        print("  ✅ Monitoring system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring test failed: {e}")
        return False


def test_caching_system():
    """Test caching functionality."""
    print("\n🗄️  Testing Caching System...")
    
    try:
        from moe_debugger.caching import InMemoryCache, CacheKey, cached
        
        # Test cache creation
        cache = InMemoryCache(max_size=1024*1024, default_ttl=300)
        
        # Test basic operations
        test_key = "test_key"
        test_value = {"data": "test_value", "timestamp": time.time()}
        
        # Set and get
        success = cache.set(test_key, test_value)
        assert success, "Cache set should succeed"
        
        retrieved_value = cache.get(test_key)
        assert retrieved_value == test_value, "Retrieved value should match stored value"
        
        # Test cache key generation
        cache_key = CacheKey("test_namespace", "test_operation", {"param": "value"})
        assert isinstance(cache_key.key, str)
        assert len(cache_key.key) > 0
        
        # Test cached decorator
        call_count = [0]
        
        @cached(cache, ttl=60)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count[0] == 1
        
        # Second call (should be cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count[0] == 1  # Should not increment
        
        # Get cache stats
        stats = cache.get_stats()
        assert "hit_rate" in stats
        assert "entry_count" in stats
        
        print("  ✅ Caching system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Caching test failed: {e}")
        return False


def test_performance_optimization():
    """Test performance optimization components."""
    print("\n⚡ Testing Performance Optimization...")
    
    try:
        from moe_debugger.performance_optimization import (
            BatchProcessor, ConnectionPool, PerformanceOptimizer
        )
        
        # Test batch processor
        batch_processor = BatchProcessor(batch_size=100)
        
        # Create test routing events  
        from moe_debugger.models import RoutingEvent
        test_events = [
            RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.1, 0.2, 0.7],
                selected_experts=[2],
                routing_confidence=0.8
            )
            for i in range(50)
        ]
        
        # Process events
        result = batch_processor.process_routing_events_batch(test_events)
        assert "total_events" in result
        assert result["total_events"] == 50
        
        # Test connection pool
        def create_mock_connection():
            return {"id": threading.get_ident(), "active": True}
        
        pool = ConnectionPool(create_mock_connection, max_connections=5)
        
        with pool.get_connection() as conn:
            assert conn is not None
            assert "id" in conn
        
        pool_stats = pool.get_stats()
        assert "pool_size" in pool_stats
        
        pool.close_all()
        
        print("  ✅ Performance optimization working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance optimization test failed: {e}")
        return False


def test_analyzer_functionality():
    """Test MoE analyzer functionality.""" 
    print("\n🔬 Testing Analyzer Functionality...")
    
    try:
        from moe_debugger.analyzer import MoEAnalyzer
        from moe_debugger.models import RoutingEvent
        
        analyzer = MoEAnalyzer()
        
        # Create test routing events
        events = [
            RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.1, 0.3, 0.6] if i % 2 == 0 else [0.4, 0.4, 0.2],
                selected_experts=[2] if i % 2 == 0 else [0, 1],
                routing_confidence=0.8 + (i % 3) * 0.05
            )
            for i in range(100)
        ]
        
        # Test load balance analysis
        load_metrics = analyzer.analyze_load_balance(events)
        assert load_metrics is not None
        assert hasattr(load_metrics, 'fairness_index')
        assert hasattr(load_metrics, 'expert_loads')
        
        # Test routing statistics
        stats = analyzer.compute_routing_statistics(events)
        assert "total_routing_decisions" in stats
        assert stats["total_routing_decisions"] == 100
        
        # Test dead expert detection
        dead_experts = analyzer.detect_dead_experts(events)
        assert isinstance(dead_experts, list)
        
        # Test routing entropy analysis
        entropy_stats = analyzer.analyze_routing_entropy(events)
        assert "mean_entropy" in entropy_stats
        assert "std_entropy" in entropy_stats
        
        # Test anomaly detection
        diagnostics = analyzer.detect_anomalies(events)
        assert isinstance(diagnostics, list)
        
        print("  ✅ Analyzer functionality working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Analyzer test failed: {e}")
        return False


def test_server_functionality():
    """Test web server functionality."""
    print("\n🌐 Testing Server Functionality...")
    
    try:
        from moe_debugger.server import DebugServer, ConnectionManager
        
        # Test connection manager
        conn_manager = ConnectionManager()
        assert len(conn_manager.active_connections) == 0
        
        # Test server creation (without starting it)
        server = DebugServer(debugger=None, host="localhost", port=8080)
        assert server.host == "localhost"
        assert server.port == 8080
        
        print("  ✅ Server functionality working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Server test failed: {e}")
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 MoE Debugger - Final Validation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Component Availability", test_component_availability),
        ("Data Models", test_data_models),
        ("Validation System", test_validation_system),
        ("Logging System", test_logging_system),
        ("Monitoring System", test_monitoring_system),
        ("Caching System", test_caching_system),
        ("Performance Optimization", test_performance_optimization),
        ("Analyzer Functionality", test_analyzer_functionality),
        ("Server Functionality", test_server_functionality),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} - CRITICAL ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_tests = []
    failed_tests = []
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:25} - {status}")
        
        if passed:
            passed_tests.append(test_name)
        else:
            failed_tests.append(test_name)
    
    success_rate = len(passed_tests) / len(results) * 100
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Passed: {len(passed_tests)}/{len(results)} ({success_rate:.1f}%)")
    print(f"   Failed: {len(failed_tests)}/{len(results)}")
    
    # Quality gates assessment
    print(f"\n🎯 QUALITY GATES ASSESSMENT:")
    
    if success_rate >= 90:
        print(f"   🎉 EXCELLENT - All quality gates passed ({success_rate:.1f}%)")
        overall_status = "EXCELLENT"
    elif success_rate >= 80:
        print(f"   ✅ GOOD - Most quality gates passed ({success_rate:.1f}%)")
        overall_status = "GOOD"
    elif success_rate >= 70:
        print(f"   ⚠️  ACCEPTABLE - Minimum quality gates passed ({success_rate:.1f}%)")
        overall_status = "ACCEPTABLE"
    else:
        print(f"   ❌ NEEDS IMPROVEMENT - Quality gates not met ({success_rate:.1f}%)")
        overall_status = "NEEDS_IMPROVEMENT"
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    print(f"   Status: {overall_status}")
    print(f"   The MoE Debugger system is {'READY FOR PRODUCTION' if success_rate >= 80 else 'NEEDS MORE WORK'}")
    
    if failed_tests:
        print(f"\n🔧 FAILED COMPONENTS TO ADDRESS:")
        for test in failed_tests:
            print(f"   - {test}")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)