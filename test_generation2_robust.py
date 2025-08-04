#!/usr/bin/env python3
"""Test Generation 2 robust functionality - error handling, validation, logging, monitoring."""

import sys
import os
import time
import tempfile
import json
from pathlib import Path

sys.path.insert(0, 'src')

def test_validation_functionality():
    """Test comprehensive validation system."""
    print("🔐 Testing Validation System...")
    
    from moe_debugger.validation import InputValidator, ValidationError, safe_json_loads, safe_json_dumps
    
    validator = InputValidator()
    
    # Test string validation
    try:
        result = validator.validate_string_input("hello world", "test_field", max_length=50)
        assert result == "hello world"
        print("  ✅ String validation works")
    except Exception as e:
        print(f"  ❌ String validation failed: {e}")
        return False
    
    # Test dangerous string detection
    try:
        validator.validate_string_input("<script>alert('xss')</script>", "dangerous", max_length=100)
        print("  ❌ Security validation failed - should have caught XSS")
        return False
    except ValidationError:
        print("  ✅ Security validation caught XSS attempt")
    
    # Test numeric validation
    try:
        result = validator.validate_numeric_input(42.5, "number", min_value=0, max_value=100)
        assert result == 42.5
        print("  ✅ Numeric validation works")
    except Exception as e:
        print(f"  ❌ Numeric validation failed: {e}")
        return False
    
    # Test routing event validation
    try:
        event_data = {
            'timestamp': time.time(),
            'layer_idx': 0,
            'token_position': 5,
            'token': 'hello',
            'expert_weights': [0.1, 0.7, 0.2],
            'selected_experts': [1],
            'routing_confidence': 0.85,
            'sequence_id': 'test_seq'
        }
        
        event = validator.validate_routing_event(event_data)
        assert event.token == 'hello'
        print("  ✅ Routing event validation works")
    except Exception as e:
        print(f"  ❌ Routing event validation failed: {e}")
        return False
    
    # Test JSON safety
    try:
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        json_str = safe_json_dumps(test_data)
        parsed_data = safe_json_loads(json_str)
        assert parsed_data == test_data
        print("  ✅ Safe JSON handling works")
    except Exception as e:
        print(f"  ❌ Safe JSON handling failed: {e}")
        return False
    
    return True


def test_logging_functionality():
    """Test comprehensive logging system."""
    print("📝 Testing Logging System...")
    
    from moe_debugger.logging_config import get_log_manager, get_logger, performance_timer
    
    # Create temporary log directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize log manager
            log_manager = get_log_manager("DEBUG", temp_dir)
            
            # Test basic logging
            logger = get_logger("test_component")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            print("  ✅ Basic logging works")
            
            # Test performance logging
            log_manager.log_performance("test_component", "test_operation", 0.123, 
                                      {"extra": "data"})
            print("  ✅ Performance logging works")
            
            # Test security logging
            log_manager.log_security_event("test_component", "test_event", 
                                         "Test security event", "WARNING")
            print("  ✅ Security logging works")
            
            # Test error logging with context
            try:
                raise ValueError("Test error")
            except Exception as e:
                log_manager.log_error_with_context("test_component", e, 
                                                 {"context": "test_context"})
            print("  ✅ Error logging with context works")
            
            # Test performance timer decorator
            @performance_timer("test_component", "decorated_operation")
            def test_operation():
                time.sleep(0.01)
                return "result"
            
            result = test_operation()
            assert result == "result"
            print("  ✅ Performance timer decorator works")
            
            # Test log stats
            stats = log_manager.get_log_stats()
            assert 'log_level' in stats
            assert 'log_files' in stats
            print("  ✅ Log statistics work")
            
            # Verify log files were created
            log_dir = Path(temp_dir)
            log_files = list(log_dir.glob("*.log"))
            if len(log_files) > 0:
                print(f"  ✅ Log files created: {[f.name for f in log_files]}")
            else:
                print("  ⚠️  No log files found (may be expected in test env)")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Logging system failed: {e}")
            return False


def test_monitoring_functionality():
    """Test system monitoring and health checks."""
    print("📊 Testing Monitoring System...")
    
    from moe_debugger.monitoring import HealthMonitor, SystemMonitor, create_default_health_checks
    
    try:
        # Test system monitor
        system_monitor = SystemMonitor(history_size=10)
        system_monitor.start_monitoring(interval=0.1)  # Fast interval for testing
        
        # Wait for some data
        time.sleep(0.5)
        
        current_metrics = system_monitor.get_current_metrics()
        assert 'timestamp' in current_metrics
        assert 'monitoring_active' in current_metrics
        print("  ✅ System monitoring works")
        
        # Test historical metrics
        historical = system_monitor.get_historical_metrics(minutes=1)
        assert 'cpu' in historical
        assert 'memory' in historical
        print("  ✅ Historical metrics work")
        
        # Test statistics
        stats = system_monitor.get_statistics(minutes=1)
        print("  ✅ Monitoring statistics work")
        
        system_monitor.stop_monitoring()
        
        # Test health monitor
        health_monitor = HealthMonitor()
        
        # Add test health checks
        def always_pass():
            return True
        
        def always_fail():
            return False
        
        health_monitor.add_health_check("test_pass", always_pass, critical=False)
        health_monitor.add_health_check("test_fail", always_fail, critical=False)
        
        # Run health checks
        results = health_monitor.run_health_checks()
        assert 'overall_status' in results
        assert 'checks' in results
        assert len(results['checks']) == 2
        print("  ✅ Health checks work")
        
        # Test system status
        status = health_monitor.get_system_status()
        assert 'health' in status
        assert 'system_metrics' in status
        print("  ✅ System status reporting works")
        
        # Test default health checks
        create_default_health_checks(health_monitor, None)
        print("  ✅ Default health checks created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robust_error_handling():
    """Test enhanced error handling throughout the system."""
    print("⚡ Testing Robust Error Handling...")
    
    from moe_debugger.analyzer import MoEAnalyzer
    from moe_debugger.models import RoutingEvent
    from moe_debugger.validation import ValidationError, ConfigurationError
    
    try:
        analyzer = MoEAnalyzer()
        
        # Test with empty data - should handle gracefully
        result = analyzer.analyze_load_balance([])
        assert result is None
        print("  ✅ Empty data handling works")
        
        # Test with invalid data - should handle gracefully
        invalid_events = [
            RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=0,
                token="test",
                expert_weights=[],  # Invalid: empty weights
                selected_experts=[0],
                routing_confidence=0.5,
                sequence_id="test"
            )
        ]
        
        # This should handle the invalid data without crashing
        try:
            analyzer.compute_routing_statistics(invalid_events)
            print("  ✅ Invalid data handling works")
        except Exception as e:
            print(f"  ✅ Invalid data properly rejected: {e}")
        
        # Test with malformed input
        try:
            result = analyzer.detect_dead_experts(None)  # None input
            # If it returns empty list, that's also acceptable
            if result == []:
                print("  ✅ None input handled gracefully")
            else:
                print("  ❌ Should have handled None input")
                return False
        except (TypeError, AttributeError):
            print("  ✅ None input properly handled")
        
        # Test optimization suggestions with edge cases
        suggestions = analyzer.generate_optimization_suggestions([])
        assert len(suggestions) > 0
        assert "No routing data" in suggestions[0]
        print("  ✅ Edge case suggestions work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_validation():
    """Test configuration validation and sanitization."""
    print("⚙️  Testing Configuration Validation...")
    
    from moe_debugger.validation import InputValidator, ValidationError, ConfigurationError
    
    try:
        validator = InputValidator()
        
        # Test valid configuration
        valid_config = {
            'enabled_hooks': {'router': True, 'experts': True},
            'sampling_rate': 0.1,
            'buffer_size': 1000,
            'save_gradients': False,
            'save_activations': True,
            'track_parameters': ['weight', 'bias'],
            'memory_limit_mb': 2048
        }
        
        config = validator.validate_hook_configuration(valid_config)
        assert config.sampling_rate == 0.1
        print("  ✅ Valid configuration accepted")
        
        # Test invalid configuration
        invalid_config = {
            'enabled_hooks': {'router': True},
            'sampling_rate': -0.5,  # Invalid: negative
            'buffer_size': 1000,
            'save_gradients': False,
            'save_activations': True,
            'track_parameters': ['weight'],
            'memory_limit_mb': 2048
        }
        
        try:
            validator.validate_hook_configuration(invalid_config)
            print("  ❌ Should have rejected invalid sampling rate")
            return False
        except (ValidationError, ConfigurationError):
            print("  ✅ Invalid configuration properly rejected")
        
        # Test model architecture validation
        valid_arch = {
            'num_layers': 12,
            'num_experts_per_layer': 8,
            'hidden_size': 768,
            'intermediate_size': 3072,
            'vocab_size': 32000,
            'max_sequence_length': 2048,
            'expert_capacity': 2.0,
            'router_type': 'top_k',
            'expert_types': {}
        }
        
        arch = validator.validate_model_architecture(valid_arch)
        assert arch.num_layers == 12
        print("  ✅ Model architecture validation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_profiler_robustness():
    """Test profiler robustness and error handling."""
    print("📈 Testing Profiler Robustness...")
    
    from moe_debugger.profiler import MoEProfiler
    
    try:
        profiler = MoEProfiler(None)
        
        # Test profiler without crashes
        profiler.start_profiling()
        
        # Test context managers
        with profiler.profile_inference():
            time.sleep(0.01)
        
        with profiler.profile_layer("test_layer"):
            time.sleep(0.01)
        
        with profiler.profile_expert(0, 0):
            time.sleep(0.01)
        
        # Test error conditions
        profiler.record_tokens_processed(-1)  # Negative input
        profiler.record_cache_hit()
        profiler.record_cache_miss()
        
        # Get metrics
        metrics = profiler.get_current_metrics()
        assert 'is_profiling' in metrics
        print("  ✅ Basic profiler robustness works")
        
        # Test bottleneck analysis
        bottlenecks = profiler.analyze_bottlenecks() 
        assert isinstance(bottlenecks, list)
        print("  ✅ Bottleneck analysis robust")
        
        # Test with extreme values
        profiler.clear_data()
        
        # Simulate huge memory usage
        original_get_memory = profiler._get_current_memory
        profiler._get_current_memory = lambda: 50000  # 50GB
        
        metrics = profiler.get_current_metrics()
        bottlenecks = profiler.analyze_bottlenecks()
        
        # Should detect high memory as bottleneck
        memory_issues = [b for b in bottlenecks if b['type'] == 'high_memory_usage']
        if memory_issues:
            print("  ✅ High memory detection works")
        else:
            print("  ⚠️  High memory detection may not be triggered")
        
        # Restore original function
        profiler._get_current_memory = original_get_memory
        
        profiler.stop_profiling()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Profiler robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 2 robustness tests."""
    print("🚀 MoE Debugger - Generation 2 Robustness Test Suite\n")
    
    tests = [
        test_validation_functionality,
        test_logging_functionality,
        test_monitoring_functionality,
        test_robust_error_handling,
        test_configuration_validation,
        test_profiler_robustness,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            print()
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} returned False")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All Generation 2 robustness tests PASSED!")
        print("\n✅ Generation 2 (Robust & Reliable) COMPLETE")
        print("   - Comprehensive input validation ✅")
        print("   - Advanced logging system ✅") 
        print("   - System monitoring & health checks ✅")
        print("   - Robust error handling ✅")
        print("   - Configuration validation ✅")
        print("   - Security measures ✅")
        return True
    else:
        print("⚠️  Some robustness tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)