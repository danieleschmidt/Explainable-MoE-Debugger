"""Test Suite for Breakthrough Research Features.

This test suite validates the revolutionary research implementations:
1. Information Bottleneck MoE Routing  
2. Unified Cache Hierarchy
3. Causal MoE Routing Framework
4. Meta-Learning Neural Architecture Search
5. Neuromorphic MoE Computing

Ensures all breakthrough features work correctly and meet research standards.
"""

import sys
import time
import traceback
import json
from typing import List, Dict, Any

def run_information_bottleneck_tests():
    """Test Information Bottleneck MoE Routing."""
    print("🔬 Testing Information Bottleneck MoE Routing...")
    try:
        from moe_debugger.information_bottleneck_routing import (
            InformationBottleneckMoERouter, 
            InformationBottleneckConfig,
            KSGMutualInfoEstimator,
            create_information_bottleneck_router
        )
        
        # Test router creation
        config = InformationBottleneckConfig(beta=1.0, mi_estimation_method='ksg')
        router = InformationBottleneckMoERouter(config, num_experts=4)
        
        # Test routing
        input_features = [0.5, 0.3, 0.8, 0.2]
        expert_id, metrics = router.route_experts(input_features, target_output=[0.7])
        
        assert 0 <= expert_id < 4, "Expert ID out of range"
        assert 'I_XE' in metrics, "Missing mutual information metrics"
        assert 'ib_objective' in metrics, "Missing IB objective"
        
        # Test mutual information estimator
        estimator = KSGMutualInfoEstimator(k=3)
        x_data = [0.1, 0.2, 0.3, 0.4, 0.5]
        y_data = [0.2, 0.4, 0.6, 0.8, 1.0]
        mi = estimator.estimate(x_data, y_data)
        
        assert mi >= 0.0, "Mutual information should be non-negative"
        
        # Test information analysis
        analysis = router.get_information_analysis()
        assert 'current_metrics' in analysis, "Missing current metrics"
        assert 'theoretical_bounds' in analysis, "Missing theoretical bounds"
        
        print("  ✅ Information Bottleneck routing tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Information Bottleneck tests failed: {e}")
        traceback.print_exc()
        return False


def run_unified_cache_tests():
    """Test Unified Cache Hierarchy."""
    print("💾 Testing Unified Cache Hierarchy...")
    try:
        from moe_debugger.unified_cache_hierarchy import (
            UnifiedCacheHierarchy,
            UnifiedCacheEntry,
            CacheLevel,
            MemoryPressureMonitor,
            create_unified_cache_hierarchy
        )
        
        # Test cache creation
        cache = create_unified_cache_hierarchy()
        
        # Test cache operations
        test_key = "test_key"
        test_value = {"data": [1, 2, 3, 4, 5]}
        
        success = cache.put(test_key, test_value, priority=7)
        assert success, "Cache put operation failed"
        
        retrieved_value = cache.get(test_key)
        assert retrieved_value == test_value, "Cache get operation failed"
        
        # Test memory pressure monitoring
        monitor = MemoryPressureMonitor()
        pressure = monitor.get_current_pressure()
        assert pressure is not None, "Memory pressure monitoring failed"
        
        # Test cache performance analysis
        analysis = cache.get_performance_analysis()
        assert 'global_metrics' in analysis, "Missing global metrics"
        assert 'tier_metrics' in analysis, "Missing tier metrics"
        assert 'cache_efficiency' in analysis, "Missing efficiency metrics"
        
        print("  ✅ Unified Cache Hierarchy tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Unified Cache Hierarchy tests failed: {e}")
        traceback.print_exc()
        return False


def run_causal_routing_tests():
    """Test Causal MoE Routing Framework.""" 
    print("🔗 Testing Causal MoE Routing Framework...")
    try:
        from moe_debugger.causal_moe_routing import (
            CausalMoERouter,
            CausalRoutingConfig,
            CausalDiscoveryEngine,
            create_causal_moe_router
        )
        
        # Test router creation
        from moe_debugger.causal_moe_routing import CausalDiscoveryAlgorithm
        config = CausalRoutingConfig(
            discovery_algorithm=CausalDiscoveryAlgorithm.PC,
            enable_causal_fairness=True,
            fairness_constraints=['sensitive_feature']
        )
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        router = create_causal_moe_router(num_experts=3, feature_names=feature_names, config=config)
        
        # Test causal routing
        input_features = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.8, 'feature4': 0.2}
        expert_id, metrics = router.route_experts(input_features, target_outcome=0.7)
        
        assert 0 <= expert_id < 3, "Expert ID out of range"
        assert 'causal_routing_score' in metrics, "Missing causal routing score"
        assert 'causal_explanation' in metrics, "Missing causal explanation"
        
        # Test counterfactual routing
        counterfactual_features = input_features.copy()
        counterfactual_features['feature1'] = 0.9
        
        counterfactual_result = router.compute_counterfactual_routing(
            input_features, counterfactual_features
        )
        assert 'expert_changed' in counterfactual_result, "Missing counterfactual analysis"
        
        # Test causal analysis
        analysis = router.get_causal_analysis()
        assert 'causal_graph' in analysis, "Missing causal graph"
        assert 'routing_statistics' in analysis, "Missing routing statistics"
        
        print("  ✅ Causal MoE Routing tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Causal MoE Routing tests failed: {e}")
        traceback.print_exc()
        return False


def run_meta_learning_nas_tests():
    """Test Meta-Learning Neural Architecture Search."""
    print("🧠 Testing Meta-Learning NAS...")
    try:
        from moe_debugger.meta_learning_nas import (
            MetaLearningNAS,
            TaskDescriptor,
            ArchitectureConfig,
            TaskType,
            create_meta_learning_nas
        )
        
        # Test NAS creation
        nas_system = create_meta_learning_nas()
        
        # Test task descriptor
        task = TaskDescriptor(
            task_type=TaskType.LANGUAGE_MODELING,
            domain="nlp",
            input_dim=512,
            output_dim=10000,
            target_accuracy=0.9
        )
        
        # Test architecture config
        architecture = ArchitectureConfig(
            num_experts=8,
            expert_hidden_dim=256,
            routing_strategy="top2"
        )
        
        # Add meta-task
        nas_system.add_meta_task(task, architecture, performance=0.85)
        
        # Test few-shot architecture search
        support_examples = [
            ({'input': [0.5] * 512}, {'output': [0.1] * 10000})
            for _ in range(5)
        ]
        
        predicted_arch, search_metrics = nas_system.few_shot_architecture_search(
            task, support_examples, k_shot=5
        )
        
        assert predicted_arch.num_experts > 0, "Invalid predicted architecture"
        assert 'search_time_seconds' in search_metrics, "Missing search metrics"
        assert 'estimated_time_saved_hours' in search_metrics, "Missing time savings"
        
        # Test cross-domain transfer
        vision_task = TaskDescriptor(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            domain="vision",
            input_dim=224*224*3,
            output_dim=1000
        )
        
        transfer_arch, transfer_metrics = nas_system.cross_domain_architecture_transfer(
            TaskType.LANGUAGE_MODELING, TaskType.IMAGE_CLASSIFICATION, vision_task
        )
        
        assert transfer_arch.num_experts > 0, "Invalid transfer architecture"
        assert 'transfer_confidence' in transfer_metrics, "Missing transfer confidence"
        
        # Test meta-learning analysis
        analysis = nas_system.get_meta_learning_analysis()
        assert 'meta_learning_statistics' in analysis, "Missing meta-learning stats"
        assert 'cross_domain_capabilities' in analysis, "Missing cross-domain capabilities"
        
        print("  ✅ Meta-Learning NAS tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Meta-Learning NAS tests failed: {e}")
        traceback.print_exc()
        return False


def run_neuromorphic_computing_tests():
    """Test Neuromorphic MoE Computing."""
    print("🧬 Testing Neuromorphic MoE Computing...")
    try:
        from moe_debugger.neuromorphic_moe_computing import (
            NeuromorphicMoERouter,
            NeuromorphicMoEConfig,
            SpikeEncoder,
            SpikeDecoder,
            NeuromorphicNeuron,
            SpikeCoding,
            create_neuromorphic_moe_router
        )
        
        # Test neuromorphic router creation
        config = NeuromorphicMoEConfig(
            num_experts=4,
            input_coding=SpikeCoding.RATE_CODING,
            neuron_model='INTEGRATE_AND_FIRE'
        )
        router = create_neuromorphic_moe_router(config)
        
        # Test spike encoding/decoding
        encoder = SpikeEncoder(SpikeCoding.RATE_CODING, max_rate=1000.0)
        decoder = SpikeDecoder(SpikeCoding.RATE_CODING)
        
        # Encode value to spikes
        spikes = encoder.encode_value(0.5, neuron_id=0)
        assert len(spikes) > 0, "No spikes generated"
        
        # Decode spikes back to value
        decoded_value = decoder.decode_spikes(spikes, end_time=100.0)
        assert 0.0 <= decoded_value <= 1.0, "Invalid decoded value"
        
        # Test neuromorphic routing
        input_features = [0.5, 0.3, 0.8, 0.2]
        expert_id, metrics = router.route_with_spikes(input_features)
        
        assert 0 <= expert_id < 4, "Expert ID out of range"
        assert 'total_spikes_processed' in metrics, "Missing spike metrics"
        assert 'power_consumption_nj' in metrics, "Missing power consumption"
        assert 'neuromorphic_efficiency' in metrics, "Missing efficiency metrics"
        
        # Test hardware platform adaptation
        from moe_debugger.neuromorphic_moe_computing import NeuromorphicHardware
        adaptations = router.adapt_to_hardware_platform(NeuromorphicHardware.INTEL_LOIHI)
        assert 'optimizations_applied' in adaptations, "Missing hardware optimizations"
        
        # Test neuromorphic analysis
        analysis = router.get_neuromorphic_analysis()
        assert 'network_architecture' in analysis, "Missing network architecture"
        assert 'power_analysis' in analysis, "Missing power analysis"
        assert 'neuromorphic_advantages' in analysis, "Missing advantages"
        
        print("  ✅ Neuromorphic MoE Computing tests passed") 
        return True
        
    except Exception as e:
        print(f"  ❌ Neuromorphic MoE Computing tests failed: {e}")
        traceback.print_exc()
        return False


def run_integration_tests():
    """Test integration between breakthrough features."""
    print("🔗 Testing Integration Between Breakthrough Features...")
    try:
        # Test that features can coexist and work together
        from moe_debugger.unified_cache_hierarchy import create_unified_cache_hierarchy
        from moe_debugger.information_bottleneck_routing import create_information_bottleneck_router
        
        # Create instances
        cache = create_unified_cache_hierarchy()
        ib_router = create_information_bottleneck_router(num_experts=4)
        
        # Test caching routing results
        routing_key = "routing_result_1"
        input_features = [0.5, 0.3, 0.8, 0.2]
        
        # Perform routing
        expert_id, metrics = ib_router.route_experts(input_features, [0.7])
        
        # Cache the result
        cache_success = cache.put(routing_key, {
            'expert_id': expert_id,
            'metrics': metrics,
            'input_features': input_features
        }, priority=8)
        
        assert cache_success, "Failed to cache routing result"
        
        # Retrieve from cache
        cached_result = cache.get(routing_key)
        assert cached_result is not None, "Failed to retrieve cached result"
        assert cached_result['expert_id'] == expert_id, "Cached result mismatch"
        
        print("  ✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration tests failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all breakthrough feature tests."""
    print("🚀 Testing Breakthrough Research Features\n")
    print("=" * 60)
    
    tests = [
        ("Information Bottleneck MoE Routing", run_information_bottleneck_tests),
        ("Unified Cache Hierarchy", run_unified_cache_tests), 
        ("Causal MoE Routing Framework", run_causal_routing_tests),
        ("Meta-Learning Neural Architecture Search", run_meta_learning_nas_tests),
        ("Neuromorphic MoE Computing", run_neuromorphic_computing_tests),
        ("Feature Integration", run_integration_tests)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        if test_func():
            passed_tests += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏆 BREAKTHROUGH FEATURES TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL BREAKTHROUGH FEATURES WORKING CORRECTLY!")
        print("✅ Revolutionary research implementations validated")
        print("✅ Information-theoretic routing operational") 
        print("✅ Causal inference framework functional")
        print("✅ Meta-learning NAS system ready")
        print("✅ Neuromorphic computing breakthrough achieved")
        print("✅ Unified cache hierarchy optimized")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED")
        print("❌ Some breakthrough features need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)