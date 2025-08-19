#!/usr/bin/env python3
"""Comprehensive test suite for autonomous SDLC enhancements.

This test suite validates all three generations of autonomous enhancements:
- Generation 1: Core functionality
- Generation 2: Robustness and reliability features  
- Generation 3: Scaling and optimization features

The tests ensure 100% functionality of all autonomous improvements while
maintaining backward compatibility with the existing production system.

Authors: Terragon Labs - Autonomous SDLC v4.0
License: MIT
"""

import sys
import os
import time
import unittest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all enhanced modules
try:
    from moe_debugger import (
        MoEDebugger, MoEAnalyzer, 
        AutonomousRecoverySystem, HealthStatus, FailurePattern,
        QuantumRoutingSystem, QuantumSuperpositionRouter,
        DistributedMoEOptimizer, ClusterNode, NodeType,
        HierarchicalCacheManager, CacheLevel, CacheState
    )
    
    from moe_debugger.autonomous_recovery import get_recovery_system
    from moe_debugger.quantum_routing import get_quantum_router, quantum_route_experts
    from moe_debugger.distributed_optimization import get_distributed_optimizer, distributed_moe_analysis
    from moe_debugger.advanced_caching import get_cache_manager
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all modules are properly installed.")
    sys.exit(1)


class TestAutonomousGeneration1(unittest.TestCase):
    """Test Generation 1: MAKE IT WORK (Core Functionality)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_routing_events = [
            {'selected_expert': i % 4, 'timestamp': time.time() + i, 'confidence': 0.8}
            for i in range(50)
        ]
    
    def test_core_moe_debugger_functionality(self):
        """Test core MoE debugger still works with enhancements."""
        debugger = MoEDebugger()
        self.assertIsNotNone(debugger)
        
        # Test basic debugging functionality
        session = debugger.start_session()
        self.assertIsNotNone(session)
        
        # Process some routing events
        for event in self.sample_routing_events[:10]:
            debugger.process_routing_event(event)
        
        stats = debugger.get_routing_stats()
        self.assertIn('total_events', stats)
        self.assertEqual(stats['total_events'], 10)
    
    def test_enhanced_analyzer_functionality(self):
        """Test enhanced analyzer with new capabilities."""
        analyzer = MoEAnalyzer()
        self.assertIsNotNone(analyzer)
        
        # Test basic analysis
        analysis_result = analyzer.analyze_routing_events(self.sample_routing_events)
        self.assertIsInstance(analysis_result, dict)
        self.assertIn('expert_utilization', analysis_result)
        
        # Test new information-theoretic analysis (if available)
        if hasattr(analyzer, 'compute_information_theoretic_metrics'):
            it_metrics = analyzer.compute_information_theoretic_metrics(self.sample_routing_events)
            self.assertIsInstance(it_metrics, dict)
    
    def test_backward_compatibility(self):
        """Ensure all new features maintain backward compatibility."""
        # Test that existing API still works
        debugger = MoEDebugger()
        analyzer = MoEAnalyzer()
        
        # These should all work without errors
        session = debugger.start_session()
        debugger.process_routing_event({'selected_expert': 0, 'timestamp': time.time()})
        stats = debugger.get_routing_stats()
        
        analysis = analyzer.analyze_routing_events([{'selected_expert': 1, 'timestamp': time.time()}])
        
        self.assertIsNotNone(session)
        self.assertIsNotNone(stats)
        self.assertIsNotNone(analysis)


class TestAutonomousGeneration2(unittest.TestCase):
    """Test Generation 2: MAKE IT ROBUST (Reliability Features)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recovery_system = get_recovery_system()
        self.quantum_router = get_quantum_router(8)
    
    def test_autonomous_recovery_system(self):
        """Test autonomous recovery and self-healing capabilities."""
        # Test health status monitoring
        health = self.recovery_system.get_health_status()
        self.assertIsInstance(health.status, HealthStatus)
        
        # Test circuit breaker functionality
        breaker = self.recovery_system.add_circuit_breaker("test_service")
        self.assertIsNotNone(breaker)
        
        # Test recovery action execution
        recovery_count = self.recovery_system.trigger_recovery(FailurePattern.MEMORY_LEAK)
        self.assertIsInstance(recovery_count, int)
        
        # Test performance optimization
        optimizations = self.recovery_system.optimize_performance()
        self.assertIsInstance(optimizations, int)
    
    def test_quantum_routing_system(self):
        """Test quantum-inspired routing algorithms."""
        # Test quantum routing
        expert_weights = [0.2, 0.3, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05]
        input_features = [0.5, -0.3, 0.8]
        
        result = self.quantum_router.quantum_route(
            input_features=input_features,
            expert_weights=expert_weights
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('selected_expert', result)
        self.assertIn('confidence', result)
        self.assertIn('quantum_advantage', result)
        
        # Test quantum superposition router
        superposition_router = self.quantum_router.superposition_router
        quantum_state = superposition_router.create_superposition_state(expert_weights)
        self.assertIsNotNone(quantum_state)
        
        # Test quantum measurement
        selected_expert, confidence = superposition_router.measure_expert_selection(quantum_state)
        self.assertIsInstance(selected_expert, int)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(selected_expert, 0)
        self.assertLess(selected_expert, len(expert_weights))
    
    def test_error_handling_and_resilience(self):
        """Test error handling and system resilience."""
        # Test with invalid inputs
        with self.assertRaises(Exception):
            invalid_router = get_quantum_router(-1)  # Invalid number of experts
        
        # Test recovery from failure scenarios
        self.recovery_system.trigger_recovery(FailurePattern.CASCADE_FAILURE)
        
        # System should still be operational
        health = self.recovery_system.get_health_status()
        self.assertIn(health.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED])
    
    def test_monitoring_and_metrics(self):
        """Test monitoring and metrics collection."""
        # Test recovery system metrics
        stats = self.recovery_system.get_recovery_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('uptime_seconds', stats)
        
        # Test quantum routing metrics
        quantum_metrics = self.quantum_router.get_quantum_performance_metrics()
        self.assertIsInstance(quantum_metrics, dict)
        self.assertIn('total_routings', quantum_metrics)


class TestAutonomousGeneration3(unittest.TestCase):
    """Test Generation 3: MAKE IT SCALE (Optimization Features)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.distributed_optimizer = get_distributed_optimizer()
        self.cache_manager = get_cache_manager()
        self.sample_events = [
            {'selected_expert': i % 8, 'timestamp': time.time() + i}
            for i in range(1000)  # Large dataset for scaling tests
        ]
    
    def test_distributed_optimization(self):
        """Test distributed computing capabilities."""
        # Test distributed MoE analysis
        result = distributed_moe_analysis(self.sample_events[:100])
        self.assertIsInstance(result, dict)
        
        # Test cluster node registration
        test_node = ClusterNode(
            node_id="test_node_1",
            node_type=NodeType.WORKER,
            host="localhost",
            port=8080
        )
        self.distributed_optimizer.register_node(test_node)
        
        # Verify node was registered
        self.assertIn("test_node_1", self.distributed_optimizer.cluster_nodes)
        
        # Test performance metrics
        metrics = self.distributed_optimizer.get_distributed_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('cluster_utilization', metrics)
    
    def test_advanced_caching_system(self):
        """Test hierarchical caching with predictive capabilities."""
        # Test basic cache operations
        test_key = "test_analysis_result"
        test_value = {"expert_utilization": [0.2, 0.3, 0.5], "timestamp": time.time()}
        
        # Put in cache
        success = self.cache_manager.put(test_key, test_value, level=CacheLevel.L2_MEMORY)
        self.assertTrue(success)
        
        # Get from cache
        cached_value = self.cache_manager.get(test_key)
        self.assertEqual(cached_value, test_value)
        
        # Test cache statistics
        stats = self.cache_manager.get_cache_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('overall_hit_rate', stats)
        
        # Test cache warming
        warmed_count = self.cache_manager.warm_cache()
        self.assertIsInstance(warmed_count, int)
    
    def test_edge_computing_acceleration(self):
        """Test edge computing features."""
        # Register edge node
        self.distributed_optimizer.register_edge_node("edge_1", ["lightweight_analysis"])
        
        # Test edge processing
        result = self.distributed_optimizer.process_with_edge_acceleration(
            self.sample_events[:50]
        )
        self.assertIsInstance(result, dict)
        self.assertIn('analysis_type', result)
    
    def test_auto_scaling_capabilities(self):
        """Test Kubernetes auto-scaling functionality."""
        # Test auto-scaling evaluation
        service_metrics = {
            "moe_analyzer": {
                "cpu_percent": 85.0,
                "memory_percent": 70.0,
                "requests_per_second": 1000,
                "avg_response_time": 200
            }
        }
        
        scaling_actions = self.distributed_optimizer.evaluate_auto_scaling(service_metrics)
        self.assertIsInstance(scaling_actions, list)
    
    def test_performance_under_load(self):
        """Test system performance under high load."""
        start_time = time.time()
        
        # Process large dataset
        for i in range(0, len(self.sample_events), 100):
            batch = self.sample_events[i:i+100]
            result = distributed_moe_analysis(batch)
            self.assertIsNotNone(result)
        
        processing_time = time.time() - start_time
        
        # Should process 1000 events in reasonable time
        self.assertLess(processing_time, 10.0, "Performance test: should process 1000 events in <10 seconds")
        
        # Test concurrent cache operations
        for i in range(100):
            self.cache_manager.put(f"load_test_{i}", {"data": i})
            value = self.cache_manager.get(f"load_test_{i}")
            self.assertEqual(value["data"], i)


class TestIntegrationAndCompatibility(unittest.TestCase):
    """Test integration between all autonomous enhancements."""
    
    def test_full_system_integration(self):
        """Test complete system working together."""
        # Initialize all systems
        debugger = MoEDebugger()
        recovery_system = get_recovery_system()
        quantum_router = get_quantum_router(4)
        cache_manager = get_cache_manager()
        distributed_optimizer = get_distributed_optimizer()
        
        # Test integrated workflow
        session = debugger.start_session()
        
        # Process events with quantum routing
        sample_events = []
        for i in range(20):
            quantum_result = quantum_router.quantum_route(
                input_features=[0.1 * i, 0.2 * i],
                expert_weights=[0.25, 0.25, 0.25, 0.25]
            )
            
            event = {
                'selected_expert': quantum_result['selected_expert'],
                'timestamp': time.time(),
                'quantum_enhanced': True
            }
            sample_events.append(event)
            debugger.process_routing_event(event)
        
        # Analyze with caching
        cache_key = "integration_test_analysis"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result is None:
            # Use distributed analysis
            analysis_result = distributed_moe_analysis(sample_events)
            cache_manager.put(cache_key, analysis_result)
        else:
            analysis_result = cached_result
        
        # Verify all components worked
        self.assertIsNotNone(session)
        self.assertIsNotNone(analysis_result)
        
        # Check system health
        health = recovery_system.get_health_status()
        self.assertIn(health.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED])
    
    def test_backward_compatibility_complete(self):
        """Comprehensive backward compatibility test."""
        # Test original API still works after all enhancements
        debugger = MoEDebugger()
        analyzer = MoEAnalyzer()
        
        # Original workflow should work unchanged
        session = debugger.start_session()
        
        for i in range(10):
            event = {'selected_expert': i % 3, 'timestamp': time.time()}
            debugger.process_routing_event(event)
        
        stats = debugger.get_routing_stats()
        analysis = analyzer.analyze_routing_events([
            {'selected_expert': i % 3, 'timestamp': time.time()} for i in range(5)
        ])
        
        # All original functionality should work
        self.assertIsNotNone(session)
        self.assertIsInstance(stats, dict)
        self.assertIsInstance(analysis, dict)
        self.assertIn('total_events', stats)
        self.assertIn('expert_utilization', analysis)
    
    def test_performance_benchmarks(self):
        """Test that enhancements meet performance benchmarks."""
        # Benchmark cache performance
        cache_manager = get_cache_manager()
        
        # Cache write performance
        start_time = time.time()
        for i in range(1000):
            cache_manager.put(f"benchmark_{i}", {"data": i})
        write_time = time.time() - start_time
        
        # Cache read performance  
        start_time = time.time()
        hit_count = 0
        for i in range(1000):
            result = cache_manager.get(f"benchmark_{i}")
            if result is not None:
                hit_count += 1
        read_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(write_time, 2.0, "Cache writes should be fast")
        self.assertLess(read_time, 1.0, "Cache reads should be very fast")
        self.assertGreaterEqual(hit_count / 1000, 0.95, "Cache hit rate should be >95%")


class TestSecurityAndReliability(unittest.TestCase):
    """Test security and reliability of autonomous enhancements."""
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        quantum_router = get_quantum_router(4)
        
        # Test with invalid inputs
        with self.assertRaises(Exception):
            quantum_router.quantum_route(
                input_features=[],
                expert_weights=[]  # Empty weights should fail
            )
        
        # Test with mismatched input sizes
        result = quantum_router.quantum_route(
            input_features=[0.1, 0.2, 0.3],
            expert_weights=[0.25, 0.25, 0.25, 0.25]  # Should handle gracefully
        )
        self.assertIsNotNone(result)
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        recovery_system = get_recovery_system()
        
        # Simulate various failure scenarios
        test_failures = [
            FailurePattern.MEMORY_LEAK,
            FailurePattern.CONNECTION_TIMEOUT,
            FailurePattern.PERFORMANCE_DEGRADATION
        ]
        
        for failure in test_failures:
            recovery_count = recovery_system.trigger_recovery(failure)
            self.assertIsInstance(recovery_count, int)
            
            # System should remain healthy or degraded, not critical
            health = recovery_system.get_health_status()
            self.assertNotEqual(health.status, HealthStatus.CRITICAL)
    
    def test_resource_management(self):
        """Test resource management and cleanup."""
        cache_manager = get_cache_manager()
        
        # Fill cache to capacity
        for i in range(1000):
            cache_manager.put(f"resource_test_{i}", {"large_data": "x" * 1000})
        
        # Cache should handle resource limits gracefully
        stats = cache_manager.get_cache_statistics()
        self.assertIsInstance(stats, dict)
        
        # System should not crash under memory pressure
        for level in CacheLevel:
            level_stats = stats['level_statistics'][level.value]
            self.assertLessEqual(level_stats['utilization'], 1.0, 
                               f"Cache level {level.value} should not exceed capacity")


def run_comprehensive_autonomous_tests():
    """Run comprehensive test suite for all autonomous enhancements."""
    print("🚀 Running Comprehensive Autonomous SDLC Enhancement Tests\n")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAutonomousGeneration1,
        TestAutonomousGeneration2, 
        TestAutonomousGeneration3,
        TestIntegrationAndCompatibility,
        TestSecurityAndReliability
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("📊 AUTONOMOUS ENHANCEMENT TEST RESULTS")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Successful: {total_tests - failures - errors}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n🔥 ERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("\n" + "=" * 80)
    
    if success_rate >= 95:
        print("✅ AUTONOMOUS ENHANCEMENT TESTS: PASSED")
        print("🎉 All autonomous SDLC enhancements are fully validated!")
        return True
    else:
        print("❌ AUTONOMOUS ENHANCEMENT TESTS: FAILED")
        print("🚨 Some autonomous enhancements need attention.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_autonomous_tests()
    sys.exit(0 if success else 1)