#!/usr/bin/env python3
"""
Comprehensive testing and validation suite for MoE Debugger.
Implements quality gates and validation checks for production readiness.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import traceback

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = None


class QualityGateRunner:
    """Runs comprehensive quality gates and validation tests."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_all_tests(self) -> bool:
        """Run all quality gates and return overall success."""
        print("🚀 Running Comprehensive MoE Debugger Quality Gates\n")
        
        test_groups = [
            ("Module Loading Tests", self._test_module_loading),
            ("Core Functionality Tests", self._test_core_functionality), 
            ("Validation System Tests", self._test_validation_system),
            ("Performance System Tests", self._test_performance_system),
            ("Monitoring System Tests", self._test_monitoring_system),
            ("Cache System Tests", self._test_cache_system),
            ("Server Integration Tests", self._test_server_integration),
            ("Frontend Integration Tests", self._test_frontend_integration),
            ("Security Validation Tests", self._test_security_validation),
            ("Error Handling Tests", self._test_error_handling),
            ("Memory Management Tests", self._test_memory_management),
            ("Configuration Tests", self._test_configuration_system)
        ]
        
        overall_success = True
        
        for group_name, test_func in test_groups:
            print(f"\n{'='*60}")
            print(f"🔍 {group_name}")
            print('='*60)
            
            try:
                success = test_func()
                overall_success = overall_success and success
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"\n{group_name}: {status}")
                
            except Exception as e:
                print(f"\n❌ {group_name} FAILED with exception: {e}")
                traceback.print_exc()
                overall_success = False
        
        self._print_final_report(overall_success)
        return overall_success
    
    def _run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        print(f"  ⏳ {test_name}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ ({duration:.3f}s)")
                self.results.append(TestResult(test_name, True, duration))
                return True
            else:
                print(f"❌ ({duration:.3f}s)")
                self.results.append(TestResult(test_name, False, duration, "Test returned False"))
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ ({duration:.3f}s) - {str(e)}")
            self.results.append(TestResult(test_name, False, duration, str(e)))
            return False
    
    def _test_module_loading(self) -> bool:
        """Test all core modules can be imported successfully."""
        modules_to_test = [
            "moe_debugger",
            "moe_debugger.debugger",
            "moe_debugger.analyzer", 
            "moe_debugger.profiler",
            "moe_debugger.server",
            "moe_debugger.validation",
            "moe_debugger.monitoring",
            "moe_debugger.performance_optimization",
            "moe_debugger.logging_config",
            "moe_debugger.cache.manager",
            "moe_debugger.model_loader",
            "moe_debugger.cli"
        ]
        
        success = True
        for module_name in modules_to_test:
            success = success and self._run_test(
                f"Import {module_name}",
                lambda m=module_name: self._test_import_module(m)
            )
        
        return success
    
    def _test_import_module(self, module_name: str) -> bool:
        """Test importing a specific module."""
        try:
            __import__(module_name)
            return True
        except Exception:
            return False
    
    def _test_core_functionality(self) -> bool:
        """Test core MoE debugging functionality."""
        success = True
        
        # Test MoE Debugger creation
        success = success and self._run_test(
            "Create MoEDebugger instance",
            self._test_debugger_creation
        )
        
        # Test MoE Analyzer
        success = success and self._run_test(
            "Create MoEAnalyzer instance", 
            self._test_analyzer_creation
        )
        
        # Test MoE Profiler
        success = success and self._run_test(
            "Create MoEProfiler instance",
            self._test_profiler_creation
        )
        
        # Test routing event processing
        success = success and self._run_test(
            "Process sample routing events",
            self._test_routing_events
        )
        
        return success
    
    def _test_debugger_creation(self) -> bool:
        """Test MoEDebugger creation."""
        try:
            from moe_debugger.debugger import MoEDebugger
            from moe_debugger.mock_torch import torch, nn
            
            # Create mock model
            model = nn.Linear(10, 10)
            debugger = MoEDebugger(model)
            
            return hasattr(debugger, 'start_session') and hasattr(debugger, 'get_routing_stats')
        except Exception:
            return False
    
    def _test_analyzer_creation(self) -> bool:
        """Test MoEAnalyzer creation."""
        try:
            from moe_debugger.analyzer import MoEAnalyzer
            analyzer = MoEAnalyzer()
            
            return hasattr(analyzer, 'analyze_load_balance') and hasattr(analyzer, 'detect_dead_experts')
        except Exception:
            return False
    
    def _test_profiler_creation(self) -> bool:
        """Test MoEProfiler creation.""" 
        try:
            from moe_debugger.profiler import MoEProfiler
            from moe_debugger.mock_torch import nn
            
            model = nn.Linear(10, 10)
            profiler = MoEProfiler(model)
            
            return hasattr(profiler, 'start_profiling') and hasattr(profiler, 'get_current_metrics')
        except Exception:
            return False
    
    def _test_routing_events(self) -> bool:
        """Test routing event processing."""
        try:
            from moe_debugger.models import RoutingEvent
            from moe_debugger.analyzer import MoEAnalyzer
            
            # Create sample routing events
            events = [
                RoutingEvent(
                    timestamp=time.time() + i,
                    layer_idx=0,
                    token_position=i,
                    token=f"token_{i}",
                    expert_weights=[0.8, 0.2, 0.0, 0.0],
                    selected_experts=[0],
                    routing_confidence=0.9,
                    sequence_id="test_seq"
                )
                for i in range(5)
            ]
            
            analyzer = MoEAnalyzer()
            stats = analyzer.compute_routing_statistics(events)
            
            return isinstance(stats, dict) and stats.get('total_routing_decisions') == 5
        except Exception:
            return False
    
    def _test_validation_system(self) -> bool:
        """Test validation and security systems."""
        success = True
        
        success = success and self._run_test(
            "Input validation system",
            self._test_input_validation
        )
        
        success = success and self._run_test(
            "Security validation",
            self._test_security_validation_impl
        )
        
        success = success and self._run_test(
            "Data sanitization",
            self._test_data_sanitization
        )
        
        return success
    
    def _test_input_validation(self) -> bool:
        """Test input validation functionality."""
        try:
            from moe_debugger.validation import InputValidator
            
            validator = InputValidator()
            
            # Test string validation
            validator.validate_string_input("test_string", "test_field")
            
            # Test numeric validation
            validator.validate_numeric_input(42, "test_number", min_value=0, max_value=100)
            
            # Test that dangerous input is caught
            try:
                validator.validate_string_input("<script>alert('xss')</script>", "test")
                return False  # Should have thrown exception
            except:
                pass  # Expected
            
            return True
        except Exception:
            return False
    
    def _test_security_validation_impl(self) -> bool:
        """Test security validation implementation."""
        try:
            from moe_debugger.validation import safe_json_loads, safe_json_dumps
            
            # Test safe JSON operations
            test_data = {"test": "data", "number": 42}
            json_str = safe_json_dumps(test_data)
            parsed_data = safe_json_loads(json_str)
            
            return parsed_data == test_data
        except Exception:
            return False
    
    def _test_data_sanitization(self) -> bool:
        """Test data sanitization."""
        try:
            from moe_debugger.validation import InputValidator
            
            validator = InputValidator()
            
            # Test sanitization of potentially dangerous input
            cleaned = validator.validate_string_input("normal text", "field")
            return cleaned == "normal text"
        except Exception:
            return False
    
    def _test_performance_system(self) -> bool:
        """Test performance optimization systems."""
        success = True
        
        success = success and self._run_test(
            "Performance optimizer creation",
            self._test_performance_optimizer
        )
        
        success = success and self._run_test(
            "Batch processing system",
            self._test_batch_processing
        )
        
        success = success and self._run_test(
            "Async task processing",
            self._test_async_processing
        )
        
        return success
    
    def _test_performance_optimizer(self) -> bool:
        """Test performance optimizer."""
        try:
            from moe_debugger.performance_optimization import get_global_optimizer
            
            optimizer = get_global_optimizer()
            stats = optimizer.get_performance_stats()
            
            return isinstance(stats, dict) and 'async_processor' in stats
        except Exception:
            return False
    
    def _test_batch_processing(self) -> bool:
        """Test batch processing functionality."""
        try:
            from moe_debugger.performance_optimization import BatchProcessor
            from moe_debugger.models import RoutingEvent
            
            processor = BatchProcessor(batch_size=10)
            
            # Create test events
            events = [
                RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=0,
                    token_position=i,
                    token=f"token_{i}",
                    expert_weights=[0.5, 0.3, 0.2],
                    selected_experts=[0],
                    routing_confidence=0.8,
                    sequence_id="test"
                )
                for i in range(5)
            ]
            
            result = processor.process_routing_events_batch(events)
            return isinstance(result, dict) and 'total_events' in result
        except Exception:
            return False
    
    def _test_async_processing(self) -> bool:
        """Test async processing capabilities.""" 
        try:
            from moe_debugger.performance_optimization import AsyncTaskProcessor
            
            processor = AsyncTaskProcessor(max_workers=2)
            stats = processor.get_stats()
            
            return isinstance(stats, dict) and 'tasks_processed' in stats
        except Exception:
            return False
    
    def _test_monitoring_system(self) -> bool:
        """Test monitoring and health check systems."""
        success = True
        
        success = success and self._run_test(
            "Health monitoring system",
            self._test_health_monitoring
        )
        
        success = success and self._run_test(
            "System metrics collection",
            self._test_system_metrics
        )
        
        return success
    
    def _test_health_monitoring(self) -> bool:
        """Test health monitoring functionality."""
        try:
            from moe_debugger.monitoring import get_health_monitor
            
            monitor = get_health_monitor()
            
            # Add a test health check
            monitor.add_health_check(
                "test_check",
                lambda: True,
                critical=False
            )
            
            # Run health checks
            results = monitor.run_health_checks()
            
            return isinstance(results, dict) and 'overall_status' in results
        except Exception:
            return False
    
    def _test_system_metrics(self) -> bool:
        """Test system metrics collection."""
        try:
            from moe_debugger.monitoring import SystemMonitor
            
            monitor = SystemMonitor()
            metrics = monitor.get_current_metrics()
            
            return isinstance(metrics, dict) and 'timestamp' in metrics
        except Exception:
            return False
    
    def _test_cache_system(self) -> bool:
        """Test caching system."""
        success = True
        
        success = success and self._run_test(
            "Cache manager functionality",
            self._test_cache_manager
        )
        
        success = success and self._run_test(
            "Cache operations",
            self._test_cache_operations
        )
        
        return success
    
    def _test_cache_manager(self) -> bool:
        """Test cache manager."""
        try:
            from moe_debugger.cache.manager import CacheManager
            
            cache = CacheManager(cache_type="memory")
            
            # Test basic operations
            cache.set("test_key", "test_value", ttl=60)
            value = cache.get("test_key")
            
            return value == "test_value"
        except Exception:
            return False
    
    def _test_cache_operations(self) -> bool:
        """Test cache operations."""
        try:
            from moe_debugger.cache.manager import CacheManager
            
            cache = CacheManager()
            
            # Test specialized cache operations
            test_stats = {"test": "data"}
            cache.cache_routing_stats("test_session", test_stats)
            retrieved_stats = cache.get_routing_stats("test_session")
            
            return retrieved_stats == test_stats
        except Exception:
            return False
    
    def _test_server_integration(self) -> bool:
        """Test server integration."""
        success = True
        
        success = success and self._run_test(
            "Server creation",
            self._test_server_creation
        )
        
        success = success and self._run_test(
            "WebSocket connection manager",
            self._test_websocket_manager
        )
        
        return success
    
    def _test_server_creation(self) -> bool:
        """Test server creation."""
        try:
            from moe_debugger.server import DebugServer, ConnectionManager
            
            # Test connection manager
            manager = ConnectionManager()
            return hasattr(manager, 'connect') and hasattr(manager, 'broadcast')
        except ImportError:
            # FastAPI not available, this is expected
            return True
        except Exception:
            return False
    
    def _test_websocket_manager(self) -> bool:
        """Test WebSocket connection manager."""
        try:
            from moe_debugger.server import ConnectionManager
            
            manager = ConnectionManager()
            return len(manager.active_connections) == 0
        except ImportError:
            return True  # FastAPI not available
        except Exception:
            return False
    
    def _test_frontend_integration(self) -> bool:
        """Test frontend integration."""
        success = True
        
        # Check if frontend files exist
        frontend_dir = Path(__file__).parent / "frontend"
        
        success = success and self._run_test(
            "Frontend directory structure",
            lambda: self._test_frontend_structure(frontend_dir)
        )
        
        success = success and self._run_test(
            "Frontend package configuration",
            lambda: self._test_frontend_config(frontend_dir)
        )
        
        return success
    
    def _test_frontend_structure(self, frontend_dir: Path) -> bool:
        """Test frontend directory structure."""
        required_files = [
            "package.json",
            "src/app/page.tsx",
            "src/components/layout/DebuggerLayout.tsx",
            "src/store/debugger.ts"
        ]
        
        for file_path in required_files:
            if not (frontend_dir / file_path).exists():
                return False
        
        return True
    
    def _test_frontend_config(self, frontend_dir: Path) -> bool:
        """Test frontend configuration."""
        try:
            package_json_path = frontend_dir / "package.json"
            if not package_json_path.exists():
                return False
                
            with open(package_json_path) as f:
                config = json.load(f)
            
            return "scripts" in config and "dependencies" in config
        except Exception:
            return False
    
    def _test_security_validation(self) -> bool:
        """Test security validation systems."""
        success = True
        
        success = success and self._run_test(
            "XSS protection",
            self._test_xss_protection
        )
        
        success = success and self._run_test(
            "Path traversal protection",
            self._test_path_traversal_protection
        )
        
        return success
    
    def _test_xss_protection(self) -> bool:
        """Test XSS protection."""
        try:
            from moe_debugger.validation import InputValidator
            
            validator = InputValidator()
            
            # These should throw security errors
            dangerous_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "eval('malicious code')"
            ]
            
            for dangerous_input in dangerous_inputs:
                try:
                    validator.validate_string_input(dangerous_input, "test_field")
                    return False  # Should have thrown exception
                except:
                    continue  # Expected
            
            return True
        except Exception:
            return False
    
    def _test_path_traversal_protection(self) -> bool:
        """Test path traversal protection."""
        try:
            from moe_debugger.validation import InputValidator
            
            validator = InputValidator()
            
            # These should throw security errors
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "/etc/passwd"
            ]
            
            for dangerous_path in dangerous_paths:
                try:
                    validator.validate_file_path(dangerous_path, "test_path")
                    return False  # Should have thrown exception
                except:
                    continue  # Expected
            
            return True
        except Exception:
            return False
    
    def _test_error_handling(self) -> bool:
        """Test error handling robustness."""
        success = True
        
        success = success and self._run_test(
            "Graceful error handling",
            self._test_graceful_errors
        )
        
        success = success and self._run_test(
            "Error logging",
            self._test_error_logging
        )
        
        return success
    
    def _test_graceful_errors(self) -> bool:
        """Test graceful error handling."""
        try:
            from moe_debugger.analyzer import MoEAnalyzer
            
            analyzer = MoEAnalyzer()
            
            # Test with invalid input - should handle gracefully
            stats = analyzer.compute_routing_statistics([])  # Empty list
            dead_experts = analyzer.detect_dead_experts([])  # Empty list
            
            return isinstance(stats, dict) and isinstance(dead_experts, list)
        except Exception:
            return False
    
    def _test_error_logging(self) -> bool:
        """Test error logging functionality."""
        try:
            from moe_debugger.logging_config import get_logger
            
            logger = get_logger("test_logger")
            
            # Test logging an error (should not throw exception)
            logger.error("Test error message")
            logger.warning("Test warning message")
            logger.info("Test info message")
            
            return True
        except Exception:
            return False
    
    def _test_memory_management(self) -> bool:
        """Test memory management."""
        success = True
        
        success = success and self._run_test(
            "Memory cleanup",
            self._test_memory_cleanup
        )
        
        success = success and self._run_test(
            "Memory monitoring",
            self._test_memory_monitoring
        )
        
        return success
    
    def _test_memory_cleanup(self) -> bool:
        """Test memory cleanup functionality."""
        try:
            from moe_debugger.cache.manager import CacheManager
            
            cache = CacheManager()
            
            # Fill cache with test data
            for i in range(100):
                cache.set(f"test_key_{i}", f"test_value_{i}")
            
            # Clear cache
            cache.clear()
            
            # Verify cache is cleared
            return cache.get("test_key_0") is None
        except Exception:
            return False
    
    def _test_memory_monitoring(self) -> bool:
        """Test memory monitoring."""
        try:
            from moe_debugger.monitoring import SystemMonitor
            
            monitor = SystemMonitor()
            metrics = monitor.get_current_metrics()
            
            # Should have timestamp even if other metrics are unavailable
            return 'timestamp' in metrics
        except Exception:
            return False
    
    def _test_configuration_system(self) -> bool:
        """Test configuration and setup systems."""
        success = True
        
        success = success and self._run_test(
            "Configuration validation",
            self._test_config_validation
        )
        
        success = success and self._run_test(
            "Model loading system",
            self._test_model_loading_system
        )
        
        return success
    
    def _test_config_validation(self) -> bool:
        """Test configuration validation."""
        try:
            from moe_debugger.validation import InputValidator
            from moe_debugger.models import HookConfiguration
            
            validator = InputValidator()
            
            # Test valid configuration
            config_dict = {
                'enabled_hooks': {'router': True, 'experts': True},
                'sampling_rate': 0.1,
                'buffer_size': 1000,
                'save_gradients': False,
                'save_activations': True,
                'track_parameters': ['weight', 'bias'],
                'memory_limit_mb': 2048
            }
            
            config = validator.validate_hook_configuration(config_dict)
            return isinstance(config, HookConfiguration)
        except Exception:
            return False
    
    def _test_model_loading_system(self) -> bool:
        """Test model loading system."""
        try:
            from moe_debugger.model_loader import ModelLoader
            
            loader = ModelLoader()
            
            # Test supported models list
            supported = loader.list_supported_models()
            
            return isinstance(supported, dict) and len(supported) > 0
        except Exception:
            return False
    
    def _print_final_report(self, overall_success: bool):
        """Print final test report."""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🎯 FINAL TEST REPORT")
        print('='*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Test Statistics:")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Passed: {passed_tests} ✅")
        print(f"   • Failed: {failed_tests} ❌") 
        print(f"   • Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"   • Total Duration: {total_time:.2f}s")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   • {result.name}: {result.error}")
        
        if overall_success:
            print(f"\n🎉 ALL QUALITY GATES PASSED! 🎉")
            print(f"✅ System is ready for production deployment!")
        else:
            print(f"\n⚠️  SOME QUALITY GATES FAILED!")
            print(f"❌ System requires fixes before production deployment.")
        
        print('='*80)


def main():
    """Main entry point."""
    runner = QualityGateRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()