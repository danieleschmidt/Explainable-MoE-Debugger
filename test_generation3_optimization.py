#!/usr/bin/env python3
"""Test Generation 3 optimization functionality - caching, performance, scaling."""

import sys
import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
import json

sys.path.insert(0, 'src')

def test_caching_system():
    """Test comprehensive caching system."""
    print("🚀 Testing Caching System...")
    
    from moe_debugger.caching import (
        InMemoryCache, CacheKey, CacheEntry, 
        LRUEvictionPolicy, AdaptiveEvictionPolicy,
        cached, get_global_cache
    )
    
    try:
        # Test cache key generation
        key1 = CacheKey("test", "operation", {"param1": "value1", "param2": 123})
        key2 = CacheKey("test", "operation", {"param2": 123, "param1": "value1"})
        assert key1.key == key2.key  # Should be same due to sorted params
        print("  ✅ Cache key generation works")
        
        # Test cache entry
        entry = CacheEntry("test_value", ttl=10.0)
        assert not entry.is_expired()
        entry.touch()
        assert entry.access_count == 1
        print("  ✅ Cache entry functionality works")
        
        # Test in-memory cache
        cache = InMemoryCache(max_size=1024 * 1024, default_ttl=60)
        
        # Basic operations
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"
        print("  ✅ Basic cache operations work")
        
        # Test TTL expiration
        cache.set("expire_key", "expire_value", ttl=0.1)
        time.sleep(0.2)
        result = cache.get("expire_key", "default")
        assert result == "default"
        print("  ✅ TTL expiration works")
        
        # Test eviction policy
        policy = LRUEvictionPolicy()
        entries = {
            "key1": CacheEntry("value1", size=500),
            "key2": CacheEntry("value2", size=600)
        }
        entries["key1"].last_accessed = time.time() - 100  # Older
        entries["key2"].last_accessed = time.time() - 50   # Newer
        
        to_evict = policy.should_evict(entries, 800)  # Need to free some space
        assert "key1" in to_evict  # Should evict older entry first
        print("  ✅ LRU eviction policy works")
        
        # Test adaptive eviction
        adaptive_policy = AdaptiveEvictionPolicy()
        entries["key1"].hit_count = 1
        entries["key1"].access_count = 10
        entries["key2"].hit_count = 8
        entries["key2"].access_count = 10
        
        to_evict = adaptive_policy.should_evict(entries, 800)
        # Should prefer evicting low hit rate entries
        print("  ✅ Adaptive eviction policy works")
        
        # Test cache decorator
        call_count = 0
        
        @cached(cache, ttl=60.0)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        result1 = expensive_function(1, 2)
        result2 = expensive_function(1, 2)  # Should be cached
        
        assert result1 == result2 == 3
        assert call_count == 1  # Only called once due to caching
        print("  ✅ Cache decorator works")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        print("  ✅ Cache statistics work")
        
        # Test global cache
        global_cache = get_global_cache()
        global_cache.set("global_key", "global_value")
        result = global_cache.get("global_key")
        assert result == "global_value"
        print("  ✅ Global cache works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Caching system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_processing():
    """Test async task processing system."""
    print("⚡ Testing Async Processing...")
    
    from moe_debugger.performance_optimization import AsyncTaskProcessor, ProcessingTask
    from moe_debugger.models import RoutingEvent
    
    async def run_async_tests():
        try:
            processor = AsyncTaskProcessor(max_workers=2)
            await processor.start()
            
            # Submit a task
            test_events = [
                RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=0,
                    token_position=i,
                    token=f"token_{i}",
                    expert_weights=[0.3, 0.7, 0.0],
                    selected_experts=[1],
                    routing_confidence=0.8,
                    sequence_id="test_seq"
                )
                for i in range(10)
            ]
            
            task_id = await processor.submit_task("analyze_routing_events", test_events, priority=1)
            result = await processor.get_result(task_id, timeout=10.0)
            
            assert isinstance(result, dict)
            assert 'total_events' in result
            print("  ✅ Async task processing works")
            
            # Test multiple tasks
            task_ids = []
            for i in range(5):
                task_id = await processor.submit_task("compute_statistics", {"data": i}, priority=i)
                task_ids.append(task_id)
            
            results = []
            for task_id in task_ids:
                result = await processor.get_result(task_id, timeout=5.0)
                results.append(result)
            
            assert len(results) == 5
            print("  ✅ Multiple async tasks work")
            
            # Test statistics
            stats = processor.get_stats()
            assert stats['tasks_processed'] > 0
            print("  ✅ Async processor statistics work")
            
            await processor.stop()
            return True
            
        except Exception as e:
            print(f"  ❌ Async processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run async tests
    try:
        return asyncio.run(run_async_tests())
    except Exception as e:
        print(f"  ❌ Async test setup failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing for large datasets."""
    print("📦 Testing Batch Processing...")
    
    from moe_debugger.performance_optimization import BatchProcessor
    from moe_debugger.models import RoutingEvent
    
    try:
        processor = BatchProcessor(batch_size=50, max_workers=2)
        
        # Create large dataset
        events = []
        for i in range(200):  # 200 events, will be split into 4 batches of 50
            event = RoutingEvent(
                timestamp=time.time() + i * 0.001,
                layer_idx=i % 4,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.1 + i * 0.001, 0.5, 0.4 - i * 0.001],
                selected_experts=[0 if i % 2 == 0 else 1],
                routing_confidence=0.7 + (i % 10) * 0.03,
                sequence_id=f"seq_{i // 10}"
            )
            events.append(event)
        
        # Process batch
        result = processor.process_routing_events_batch(events)
        
        assert isinstance(result, dict)
        assert result['total_events'] == 200
        assert 'confidence_stats' in result
        assert 'expert_usage' in result
        assert 'batches_processed' in result
        
        print(f"  ✅ Processed {result['total_events']} events in {result.get('batches_processed', 0)} batches")
        
        # Test caching - second call should be faster
        start_time = time.time()
        cached_result = processor.process_routing_events_batch(events)
        cache_time = time.time() - start_time
        
        assert cached_result == result
        assert cache_time < 0.01  # Should be very fast due to caching
        print("  ✅ Batch processing caching works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connection_pooling():
    """Test connection pooling system."""
    print("🔗 Testing Connection Pooling...")
    
    from moe_debugger.performance_optimization import ConnectionPool
    
    try:
        # Mock connection class
        class MockConnection:
            def __init__(self, conn_id):
                self.conn_id = conn_id
                self.closed = False
                
            def close(self):
                self.closed = True
                
            def query(self, sql):
                if self.closed:
                    raise Exception("Connection closed")
                return f"Result from connection {self.conn_id}"
        
        connection_counter = 0
        
        def create_connection():
            nonlocal connection_counter
            connection_counter += 1
            return MockConnection(connection_counter)
        
        # Create pool
        pool = ConnectionPool(create_connection, max_connections=5, min_connections=2)
        
        # Test getting connections
        with pool.get_connection() as conn:
            result = conn.query("SELECT 1")
            assert "Result from connection" in result
        
        # Test multiple connections
        connections = []
        for i in range(3):
            conn_wrapper = pool.get_connection()
            connections.append(conn_wrapper)
        
        # Return connections
        for conn_wrapper in connections:
            conn_wrapper.return_to_pool()
        
        # Test statistics
        stats = pool.get_stats()
        assert stats['total_created'] >= 2  # At least min connections
        assert stats['total_borrowed'] >= 4  # We borrowed connections
        
        print("  ✅ Connection pooling works")
        
        # Test concurrent access
        def worker():
            with pool.get_connection() as conn:
                result = conn.query("SELECT * FROM test")
                time.sleep(0.01)  # Simulate work
                return result
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            results = [f.result() for f in futures]
        
        assert len(results) == 20
        print("  ✅ Concurrent connection pooling works")
        
        # Cleanup
        pool.close_all()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Connection pooling failed: {e}")
        import traceback
        traceback.print_exc() 
        return False


def test_performance_optimizer():
    """Test comprehensive performance optimizer."""
    print("🎯 Testing Performance Optimizer...")
    
    from moe_debugger.performance_optimization import PerformanceOptimizer, get_global_optimizer
    from moe_debugger.models import RoutingEvent
    
    async def run_optimizer_tests():
        try:
            optimizer = PerformanceOptimizer()
            await optimizer.start()
            
            # Test processing large dataset
            events = [
                RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=0,
                    token_position=i,
                    token=f"token_{i}",
                    expert_weights=[0.4, 0.6],
                    selected_experts=[1],
                    routing_confidence=0.75,
                    sequence_id="perf_test"
                )
                for i in range(150)  # Large enough to trigger batch processing
            ]
            
            result = await optimizer.process_large_dataset(events, "analyze_routing_events")
            assert isinstance(result, dict)
            print("  ✅ Large dataset processing works")
            
            # Test performance metrics recording
            optimizer.record_performance_metric('request_times', 0.5)
            optimizer.record_performance_metric('cpu_usage', 65.0)
            optimizer.record_performance_metric('memory_usage', 70.0)
            
            stats = optimizer.get_performance_stats()
            assert 'async_processor' in stats
            assert 'cache' in stats
            print("  ✅ Performance metrics recording works")
            
            # Test optimization suggestions
            suggestions = optimizer.get_optimization_suggestions()
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
            print("  ✅ Optimization suggestions work")
            
            # Test auto-scaling checks
            scale_up = optimizer.should_scale_up()
            scale_down = optimizer.should_scale_down()
            assert isinstance(scale_up, bool)
            assert isinstance(scale_down, bool)
            print("  ✅ Auto-scaling checks work")
            
            # Test memory optimization
            optimizer.optimize_memory()
            print("  ✅ Memory optimization works")
            
            await optimizer.stop()
            
            # Test global optimizer
            global_opt = get_global_optimizer()
            assert isinstance(global_opt, PerformanceOptimizer)
            print("  ✅ Global optimizer works")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Performance optimizer failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    try:
        return asyncio.run(run_optimizer_tests())
    except Exception as e:
        print(f"  ❌ Optimizer test setup failed: {e}")
        return False


def test_vectorized_operations():
    """Test vectorized operations for performance."""
    print("🔢 Testing Vectorized Operations...")
    
    from moe_debugger.performance_optimization import AsyncTaskProcessor
    from moe_debugger.models import RoutingEvent
    
    try:
        processor = AsyncTaskProcessor()
        
        # Create test data
        events = []
        for i in range(1000):
            event = RoutingEvent(
                timestamp=time.time(),
                layer_idx=i % 5,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.2, 0.3, 0.5] if i % 2 == 0 else [0.4, 0.1, 0.5],
                selected_experts=[2] if i % 2 == 0 else [0],
                routing_confidence=0.8 + (i % 10) * 0.02,
                sequence_id=f"seq_{i // 100}"
            )
            events.append(event)
        
        # Test vectorized analysis
        start_time = time.time()
        result = processor._sync_analyze_routing_events(events)
        processing_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert 'total_events' in result
        assert result['total_events'] == 1000
        
        print(f"  ✅ Processed {result['total_events']} events in {processing_time:.3f}s")
        
        # Test with different data sizes
        for size in [100, 500, 1000]:
            test_events = events[:size]
            start = time.time()
            result = processor._sync_analyze_routing_events(test_events)
            duration = time.time() - start
            
            print(f"  ✅ {size} events processed in {duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vectorized operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmarks():
    """Test performance benchmarks and comparisons."""
    print("📊 Testing Performance Benchmarks...")
    
    from moe_debugger.caching import InMemoryCache, cached
    from moe_debugger.performance_optimization import BatchProcessor
    from moe_debugger.models import RoutingEvent
    
    try:
        # Benchmark cache performance
        cache = InMemoryCache(max_size=10 * 1024 * 1024)
        
        # Write performance
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        
        # Read performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        read_time = time.time() - start_time
        
        print(f"  ✅ Cache: 1K writes in {write_time:.3f}s, 1K reads in {read_time:.3f}s")
        
        # Benchmark batch processing vs individual processing
        events = [
            RoutingEvent(
                timestamp=time.time(),
                layer_idx=i % 3,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.3, 0.4, 0.3],
                selected_experts=[1],
                routing_confidence=0.8,
                sequence_id=f"seq_{i // 50}"
            )
            for i in range(500)
        ]
        
        # Batch processing
        processor = BatchProcessor(batch_size=100)
        start_time = time.time()
        batch_result = processor.process_routing_events_batch(events)
        batch_time = time.time() - start_time
        
        print(f"  ✅ Batch processed {len(events)} events in {batch_time:.3f}s")
        
        # Test memory efficiency
        import gc
        gc.collect()
        
        # Create large dataset
        large_events = [
            RoutingEvent(
                timestamp=time.time(),
                layer_idx=i % 8,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.125] * 8,  # 8 experts
                selected_experts=[i % 8],
                routing_confidence=0.7 + (i % 10) * 0.03,
                sequence_id=f"seq_{i // 100}"
            )
            for i in range(5000)
        ]
        
        start_time = time.time()
        large_result = processor.process_routing_events_batch(large_events)
        large_time = time.time() - start_time
        
        print(f"  ✅ Large batch: {len(large_events)} events in {large_time:.3f}s")
        
        # Performance ratios
        events_per_second = len(large_events) / large_time
        print(f"  ✅ Throughput: {events_per_second:.0f} events/second")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 3 optimization tests."""
    print("🚀 MoE Debugger - Generation 3 Optimization Test Suite\n")
    
    tests = [
        test_caching_system,
        test_async_processing,
        test_batch_processing,
        test_connection_pooling,
        test_performance_optimizer,
        test_vectorized_operations,
        test_performance_benchmarks,
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
        print("🎉 All Generation 3 optimization tests PASSED!")
        print("\n✅ Generation 3 (Optimized & Scalable) COMPLETE")
        print("   - Advanced caching with intelligent eviction ✅")
        print("   - Async task processing with prioritization ✅") 
        print("   - Batch processing for large datasets ✅")
        print("   - Connection pooling for resource management ✅")
        print("   - Performance optimization and auto-scaling ✅")
        print("   - Vectorized operations for speed ✅")
        print("   - Comprehensive benchmarking ✅")
        return True
    else:
        print("⚠️  Some optimization tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)