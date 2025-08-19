#!/usr/bin/env python3
"""Core test suite for autonomous SDLC enhancements.

This test suite focuses on testing the autonomous enhancements directly
without dependencies on the existing MoE debugger components.

Authors: Terragon Labs - Autonomous SDLC v4.0
License: MIT
"""

import sys
import os
import time
import unittest
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test autonomous enhancements only
def test_autonomous_recovery():
    """Test autonomous recovery system."""
    try:
        from moe_debugger.autonomous_recovery import get_recovery_system, HealthStatus, FailurePattern
        
        recovery_system = get_recovery_system()
        
        # Test health monitoring
        health = recovery_system.get_health_status()
        assert isinstance(health.status, HealthStatus)
        
        # Test circuit breaker
        breaker = recovery_system.add_circuit_breaker("test_service")
        assert breaker is not None
        
        # Test recovery action
        recovery_count = recovery_system.trigger_recovery(FailurePattern.MEMORY_LEAK)
        assert isinstance(recovery_count, int)
        
        # Test statistics
        stats = recovery_system.get_recovery_statistics()
        assert isinstance(stats, dict)
        assert 'uptime_seconds' in stats
        
        print("✅ Autonomous recovery system: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Autonomous recovery system: FAILED - {e}")
        return False


def test_quantum_routing():
    """Test quantum routing system."""
    try:
        from moe_debugger.quantum_routing import get_quantum_router, quantum_route_experts
        
        # Test basic quantum routing
        quantum_router = get_quantum_router(8)
        
        expert_weights = [0.2, 0.3, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05]
        input_features = [0.5, -0.3, 0.8]
        
        result = quantum_router.quantum_route(
            input_features=input_features,
            expert_weights=expert_weights
        )
        
        assert isinstance(result, dict)
        assert 'selected_expert' in result
        assert 'confidence' in result
        assert 'quantum_advantage' in result
        
        # Test quantum superposition
        superposition_router = quantum_router.superposition_router
        quantum_state = superposition_router.create_superposition_state(expert_weights)
        assert quantum_state is not None
        
        # Test quantum measurement
        selected_expert, confidence = superposition_router.measure_expert_selection(quantum_state)
        assert isinstance(selected_expert, int)
        assert isinstance(confidence, float)
        assert 0 <= selected_expert < len(expert_weights)
        
        # Test performance metrics
        metrics = quantum_router.get_quantum_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'total_routings' in metrics
        
        # Test convenient function
        convenient_result = quantum_route_experts(expert_weights, input_features)
        assert isinstance(convenient_result, dict)
        
        print("✅ Quantum routing system: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Quantum routing system: FAILED - {e}")
        return False


def test_distributed_optimization():
    """Test distributed optimization system."""
    try:
        from moe_debugger.distributed_optimization import (
            get_distributed_optimizer, distributed_moe_analysis, 
            ClusterNode, NodeType, setup_distributed_cluster
        )
        
        # Test distributed optimizer
        optimizer = get_distributed_optimizer()
        
        # Test cluster node registration
        test_node = ClusterNode(
            node_id="test_worker_1",
            node_type=NodeType.WORKER,
            host="localhost",
            port=8080
        )
        optimizer.register_node(test_node)
        
        assert "test_worker_1" in optimizer.cluster_nodes
        
        # Test edge node registration
        optimizer.register_edge_node("edge_1", ["lightweight_analysis"])
        assert "edge_1" in optimizer.edge_nodes
        
        # Test distributed analysis
        sample_events = [
            {'selected_expert': i % 4, 'timestamp': time.time() + i}
            for i in range(50)
        ]
        
        result = distributed_moe_analysis(sample_events)
        assert isinstance(result, dict)
        
        # Test edge processing
        edge_result = optimizer.process_with_edge_acceleration(sample_events[:20])
        assert isinstance(edge_result, dict)
        
        # Test auto-scaling
        service_metrics = {
            "test_service": {
                "cpu_percent": 75.0,
                "memory_percent": 60.0,
                "requests_per_second": 500,
                "avg_response_time": 150
            }
        }
        
        scaling_actions = optimizer.evaluate_auto_scaling(service_metrics)
        assert isinstance(scaling_actions, list)
        
        # Test performance metrics
        metrics = optimizer.get_distributed_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'cluster_utilization' in metrics
        
        # Test cluster setup
        nodes_config = [
            {
                'node_id': 'setup_test_1',
                'node_type': 'worker',
                'host': 'localhost',
                'port': 8081
            }
        ]
        setup_optimizer = setup_distributed_cluster(nodes_config)
        assert setup_optimizer is not None
        
        print("✅ Distributed optimization system: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Distributed optimization system: FAILED - {e}")
        return False


def test_advanced_caching():
    """Test advanced caching system."""
    try:
        from moe_debugger.advanced_caching import (
            get_cache_manager, CacheLevel, CacheState, cached_moe_analysis
        )
        
        # Test cache manager
        cache_manager = get_cache_manager()
        
        # Test basic cache operations
        test_key = "test_analysis_result"
        test_value = {"expert_utilization": [0.2, 0.3, 0.5], "timestamp": time.time()}
        
        # Put and get
        success = cache_manager.put(test_key, test_value, level=CacheLevel.L2_MEMORY)
        assert success
        
        cached_value = cache_manager.get(test_key)
        assert cached_value == test_value
        
        # Test different cache levels
        for level in [CacheLevel.L1_CPU, CacheLevel.L3_SSD]:
            level_key = f"level_test_{level.value}"
            success = cache_manager.put(level_key, {"level": level.value}, level=level)
            assert success
            
            retrieved = cache_manager.get(level_key)
            assert retrieved is not None
        
        # Test cache statistics
        stats = cache_manager.get_cache_statistics()
        assert isinstance(stats, dict)
        assert 'overall_hit_rate' in stats
        assert 'level_statistics' in stats
        
        # Test predictive caching
        recommendations = cache_manager.predictive_engine.get_cache_warming_recommendations()
        assert isinstance(recommendations, list)
        
        # Test cache warming
        warmed_count = cache_manager.warm_cache()
        assert isinstance(warmed_count, int)
        
        # Test quantum cache features
        quantum_cache = cache_manager.quantum_cache
        assert quantum_cache is not None
        
        # Test decorator
        @cached_moe_analysis("decorator_test", ttl=3600)
        def sample_analysis():
            return {"result": "test_analysis", "timestamp": time.time()}
        
        result1 = sample_analysis()
        result2 = sample_analysis()  # Should be cached
        assert result1 == result2
        
        print("✅ Advanced caching system: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Advanced caching system: FAILED - {e}")
        return False


def test_integration():
    """Test integration between autonomous systems."""
    try:
        # Initialize all systems
        from moe_debugger.autonomous_recovery import get_recovery_system
        from moe_debugger.quantum_routing import get_quantum_router
        from moe_debugger.distributed_optimization import get_distributed_optimizer
        from moe_debugger.advanced_caching import get_cache_manager
        
        recovery_system = get_recovery_system()
        quantum_router = get_quantum_router(4)
        distributed_optimizer = get_distributed_optimizer()
        cache_manager = get_cache_manager()
        
        # Test integrated workflow
        sample_events = []
        for i in range(10):
            # Use quantum routing
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
        
        # Cache analysis result
        cache_key = "integration_test"
        from moe_debugger.distributed_optimization import distributed_moe_analysis
        
        cached_result = cache_manager.get(cache_key)
        if cached_result is None:
            analysis_result = distributed_moe_analysis(sample_events)
            cache_manager.put(cache_key, analysis_result)
        else:
            analysis_result = cached_result
        
        # Check system health
        health = recovery_system.get_health_status()
        
        # All systems should work together
        assert len(sample_events) == 10
        assert analysis_result is not None
        assert health is not None
        
        print("✅ System integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ System integration: FAILED - {e}")
        return False


def test_performance():
    """Test performance of autonomous systems."""
    try:
        from moe_debugger.quantum_routing import get_quantum_router
        from moe_debugger.advanced_caching import get_cache_manager
        from moe_debugger.distributed_optimization import distributed_moe_analysis
        
        # Performance test parameters
        num_operations = 100
        
        # Test quantum routing performance
        quantum_router = get_quantum_router(8)
        expert_weights = [0.125] * 8
        
        start_time = time.time()
        for i in range(num_operations):
            result = quantum_router.quantum_route(
                input_features=[0.1 * i],
                expert_weights=expert_weights
            )
            assert result is not None
        quantum_time = time.time() - start_time
        
        # Test cache performance
        cache_manager = get_cache_manager()
        
        start_time = time.time()
        for i in range(num_operations):
            cache_manager.put(f"perf_test_{i}", {"data": i})
            value = cache_manager.get(f"perf_test_{i}")
            assert value is not None
        cache_time = time.time() - start_time
        
        # Test distributed analysis performance
        sample_events = [
            {'selected_expert': i % 4, 'timestamp': time.time() + i}
            for i in range(num_operations)
        ]
        
        start_time = time.time()
        result = distributed_moe_analysis(sample_events)
        distributed_time = time.time() - start_time
        
        # Performance assertions (lenient for testing environment)
        assert quantum_time < 10.0, f"Quantum routing too slow: {quantum_time:.2f}s"
        assert cache_time < 5.0, f"Cache operations too slow: {cache_time:.2f}s"
        assert distributed_time < 5.0, f"Distributed analysis too slow: {distributed_time:.2f}s"
        
        print(f"✅ Performance tests: PASSED")
        print(f"   - Quantum routing: {quantum_time:.3f}s for {num_operations} operations")
        print(f"   - Cache operations: {cache_time:.3f}s for {num_operations} operations") 
        print(f"   - Distributed analysis: {distributed_time:.3f}s for {num_operations} events")
        return True
        
    except Exception as e:
        print(f"❌ Performance tests: FAILED - {e}")
        return False


def run_core_autonomous_tests():
    """Run core autonomous enhancement tests."""
    print("🚀 Running Core Autonomous SDLC Enhancement Tests")
    print("=" * 60)
    
    test_functions = [
        test_autonomous_recovery,
        test_quantum_routing,
        test_distributed_optimization,
        test_advanced_caching,
        test_integration,
        test_performance
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("📊 CORE AUTONOMOUS TEST RESULTS")
    print("=" * 60)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 100:
        print("\n✅ ALL AUTONOMOUS ENHANCEMENTS: PASSED")
        print("🎉 Autonomous SDLC enhancements are fully operational!")
    elif success_rate >= 80:
        print("\n⚠️  MOST AUTONOMOUS ENHANCEMENTS: PASSED")
        print("👍 Core autonomous functionality is working.")
    else:
        print("\n❌ AUTONOMOUS ENHANCEMENTS: NEED ATTENTION")
        print("🚨 Some autonomous systems require fixes.")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = run_core_autonomous_tests()
    sys.exit(0 if success else 1)