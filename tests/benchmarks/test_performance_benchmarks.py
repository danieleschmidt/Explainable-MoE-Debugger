"""
Performance benchmark tests for progressive quality gates.
"""

import pytest
import time
import asyncio
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import json
from pathlib import Path

# Import the actual modules
from moe_debugger import MoEDebugger
from moe_debugger.debugger import EnhancedMoEDebugger
from moe_debugger.analyzer import MoEAnalyzer
from moe_debugger.cache.manager import CacheManager
from moe_debugger.performance_optimization import PerformanceOptimizer


class BenchmarkMetrics:
    """Collect and track benchmark metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.end_time = time.time()
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage."""
        process = psutil.Process()
        while self._monitoring:
            try:
                self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                self.cpu_usage.append(process.cpu_percent())
                time.sleep(0.1)
            except psutil.NoSuchProcess:
                break
    
    @property
    def duration(self):
        """Get benchmark duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def max_memory_mb(self):
        """Get maximum memory usage."""
        return max(self.memory_usage) if self.memory_usage else 0
    
    @property
    def avg_cpu_percent(self):
        """Get average CPU usage."""
        return sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0


@pytest.fixture
def mock_model():
    """Create a mock MoE model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.num_experts = 8
    model.config.num_local_experts = 8
    model.config.router_jitter_noise = 0.01
    return model


@pytest.fixture
def debugger(mock_model):
    """Create debugger instance for benchmarking."""
    return MoEDebugger(model=mock_model)


@pytest.fixture
def enhanced_debugger(mock_model):
    """Create enhanced debugger instance for benchmarking."""
    return EnhancedMoEDebugger(model=mock_model)


class TestCoreFunctionalityBenchmarks:
    """Benchmark core MoE debugging functionality."""
    
    def test_debugger_initialization_performance(self, benchmark, mock_model):
        """Benchmark debugger initialization time."""
        def create_debugger():
            return MoEDebugger(model=mock_model)
        
        debugger = benchmark(create_debugger)
        assert debugger is not None
        
        # Custom metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "initialization_complexity": "O(1)",
                "memory_overhead_mb": 10.5  # Estimated
            }
        })
    
    def test_routing_event_processing_performance(self, benchmark, debugger):
        """Benchmark routing event processing speed."""
        # Generate test routing data
        routing_events = [
            {
                "expert_id": i % 8,
                "token_id": i,
                "routing_weight": 0.1 + (i % 10) * 0.1,
                "layer_id": i % 4,
                "sequence_position": i % 128
            }
            for i in range(1000)
        ]
        
        def process_routing_events():
            for event in routing_events:
                debugger.process_routing_event(event)
        
        result = benchmark(process_routing_events)
        
        # Custom performance metrics
        events_per_second = len(routing_events) / benchmark.stats.mean
        benchmark.extra_info.update({
            "custom_metrics": {
                "events_processed": len(routing_events),
                "events_per_second": events_per_second,
                "throughput_target_met": events_per_second >= 10000
            }
        })
    
    def test_expert_utilization_analysis_performance(self, benchmark, enhanced_debugger):
        """Benchmark expert utilization analysis."""
        # Mock routing data
        with patch.object(enhanced_debugger, 'routing_data') as mock_routing_data:
            mock_routing_data.return_value = [
                {"expert_id": i % 8, "routing_weight": 0.1 + (i % 10) * 0.1}
                for i in range(5000)
            ]
            
            def analyze_expert_utilization():
                return enhanced_debugger.analyze_expert_utilization()
            
            result = benchmark(analyze_expert_utilization)
            assert result is not None
            
            # Performance metrics
            benchmark.extra_info.update({
                "custom_metrics": {
                    "analysis_data_points": 5000,
                    "analysis_time_target_met": benchmark.stats.mean < 0.1  # < 100ms
                }
            })


class TestCachingPerformanceBenchmarks:
    """Benchmark caching system performance."""
    
    def test_cache_write_performance(self, benchmark):
        """Benchmark cache write operations."""
        cache_manager = CacheManager()
        
        test_data = {
            f"key_{i}": {
                "routing_data": [j for j in range(100)],
                "timestamp": time.time(),
                "expert_utilization": {f"expert_{k}": k * 0.1 for k in range(8)}
            }
            for i in range(100)
        }
        
        def write_cache_data():
            for key, data in test_data.items():
                cache_manager.set(key, data)
        
        benchmark(write_cache_data)
        
        # Cache performance metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "cache_operations": len(test_data),
                "cache_throughput_target_met": benchmark.stats.mean < 0.5  # < 500ms
            }
        })
    
    def test_cache_read_performance(self, benchmark):
        """Benchmark cache read operations."""
        cache_manager = CacheManager()
        
        # Pre-populate cache
        for i in range(1000):
            cache_manager.set(f"test_key_{i}", {"data": f"value_{i}"})
        
        keys_to_read = [f"test_key_{i}" for i in range(0, 1000, 10)]  # 100 keys
        
        def read_cache_data():
            results = []
            for key in keys_to_read:
                result = cache_manager.get(key)
                results.append(result)
            return results
        
        results = benchmark(read_cache_data)
        assert len(results) == len(keys_to_read)
        
        # Read performance metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "cache_reads": len(keys_to_read),
                "cache_hit_rate": sum(1 for r in results if r is not None) / len(results),
                "read_throughput_target_met": benchmark.stats.mean < 0.1  # < 100ms
            }
        })


class TestConcurrencyBenchmarks:
    """Benchmark concurrent operations performance."""
    
    def test_concurrent_routing_processing(self, benchmark, enhanced_debugger):
        """Benchmark concurrent routing event processing."""
        
        def generate_routing_events(num_events):
            return [
                {
                    "expert_id": i % 8,
                    "token_id": i,
                    "routing_weight": 0.1 + (i % 10) * 0.1,
                    "timestamp": time.time()
                }
                for i in range(num_events)
            ]
        
        def process_events_concurrently():
            events = generate_routing_events(1000)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Split events into chunks for concurrent processing
                chunk_size = len(events) // 4
                chunks = [
                    events[i:i + chunk_size] 
                    for i in range(0, len(events), chunk_size)
                ]
                
                futures = []
                for chunk in chunks:
                    future = executor.submit(
                        lambda chunk: [enhanced_debugger.process_routing_event(event) for event in chunk],
                        chunk
                    )
                    futures.append(future)
                
                # Wait for all chunks to complete
                results = [future.result() for future in futures]
                return sum(len(chunk_result) for chunk_result in results)
        
        total_processed = benchmark(process_events_concurrently)
        assert total_processed >= 1000
        
        # Concurrency metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "concurrent_workers": 4,
                "events_processed": total_processed,
                "concurrency_efficiency": total_processed / 1000,
                "concurrent_throughput_target_met": benchmark.stats.mean < 1.0  # < 1 second
            }
        })
    
    @pytest.mark.asyncio
    async def test_async_operations_performance(self, benchmark):
        """Benchmark asynchronous operations."""
        
        async def async_processing_simulation():
            """Simulate async processing of routing data."""
            tasks = []
            
            for i in range(100):
                async def process_item(item_id):
                    # Simulate async processing
                    await asyncio.sleep(0.001)  # 1ms delay
                    return {"processed_id": item_id, "result": item_id * 2}
                
                tasks.append(process_item(i))
            
            results = await asyncio.gather(*tasks)
            return results
        
        def run_async_benchmark():
            return asyncio.run(async_processing_simulation())
        
        results = benchmark(run_async_benchmark)
        assert len(results) == 100
        
        # Async performance metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "async_tasks": len(results),
                "async_efficiency": len(results) / benchmark.stats.mean,
                "async_target_met": benchmark.stats.mean < 2.0  # < 2 seconds
            }
        })


class TestMemoryPerformanceBenchmarks:
    """Benchmark memory usage and efficiency."""
    
    def test_memory_usage_under_load(self, benchmark, enhanced_debugger):
        """Benchmark memory usage under high load."""
        
        def memory_stress_test():
            metrics = BenchmarkMetrics()
            metrics.start_monitoring()
            
            try:
                # Generate large dataset
                large_dataset = []
                for i in range(10000):
                    routing_event = {
                        "expert_id": i % 8,
                        "token_id": i,
                        "routing_weight": 0.1 + (i % 10) * 0.1,
                        "layer_id": i % 4,
                        "sequence_position": i % 128,
                        "additional_data": [j for j in range(10)]  # Extra data
                    }
                    large_dataset.append(routing_event)
                    enhanced_debugger.process_routing_event(routing_event)
                
                # Force some analysis
                enhanced_debugger.analyze_expert_utilization()
                enhanced_debugger.get_performance_metrics()
                
                return len(large_dataset)
                
            finally:
                metrics.stop_monitoring()
                
                # Store memory metrics
                benchmark.extra_info.update({
                    "custom_metrics": {
                        "max_memory_mb": metrics.max_memory_mb,
                        "avg_cpu_percent": metrics.avg_cpu_percent,
                        "memory_efficient": metrics.max_memory_mb < 500,  # < 500MB
                        "duration_seconds": metrics.duration
                    }
                })
        
        result = benchmark(memory_stress_test)
        assert result == 10000


class TestScalabilityBenchmarks:
    """Benchmark scalability characteristics."""
    
    @pytest.mark.parametrize("num_experts", [4, 8, 16, 32])
    def test_scaling_with_expert_count(self, benchmark, num_experts):
        """Benchmark performance scaling with number of experts."""
        
        # Create mock model with variable expert count
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.num_experts = num_experts
        mock_model.config.num_local_experts = num_experts
        
        def scaling_test():
            debugger = EnhancedMoEDebugger(model=mock_model)
            
            # Generate events for all experts
            events = [
                {
                    "expert_id": i % num_experts,
                    "token_id": i,
                    "routing_weight": 0.1 + (i % 10) * 0.1
                }
                for i in range(num_experts * 100)  # 100 events per expert
            ]
            
            for event in events:
                debugger.process_routing_event(event)
            
            # Analyze utilization
            utilization = debugger.analyze_expert_utilization()
            return len(events)
        
        events_processed = benchmark(scaling_test)
        
        # Scaling metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "num_experts": num_experts,
                "events_processed": events_processed,
                "events_per_expert": events_processed / num_experts,
                "scaling_efficiency": 1000 / (benchmark.stats.mean * num_experts),
                "linear_scaling": benchmark.stats.mean < (num_experts * 0.01)  # Should scale linearly
            }
        })
    
    @pytest.mark.parametrize("data_size", [1000, 5000, 10000, 50000])
    def test_scaling_with_data_size(self, benchmark, enhanced_debugger, data_size):
        """Benchmark performance scaling with data size."""
        
        def data_scaling_test():
            events = [
                {
                    "expert_id": i % 8,
                    "token_id": i,
                    "routing_weight": 0.1 + (i % 10) * 0.1,
                    "layer_id": i % 4
                }
                for i in range(data_size)
            ]
            
            for event in events:
                enhanced_debugger.process_routing_event(event)
            
            # Trigger analysis
            enhanced_debugger.analyze_expert_utilization()
            return data_size
        
        processed_count = benchmark(data_scaling_test)
        assert processed_count == data_size
        
        # Data scaling metrics
        throughput = data_size / benchmark.stats.mean
        benchmark.extra_info.update({
            "custom_metrics": {
                "data_size": data_size,
                "throughput_events_per_second": throughput,
                "sub_linear_scaling": throughput > data_size * 0.8,  # Should maintain good throughput
                "processing_time_per_event_ms": (benchmark.stats.mean / data_size) * 1000
            }
        })


class TestRealWorldScenarioBenchmarks:
    """Benchmark realistic usage scenarios."""
    
    def test_typical_debugging_session(self, benchmark, enhanced_debugger):
        """Benchmark a typical debugging session workflow."""
        
        def debugging_session():
            # 1. Start session
            session_id = enhanced_debugger.start_session()
            
            # 2. Process routing events (simulating model inference)
            for batch in range(10):  # 10 batches
                events = [
                    {
                        "expert_id": i % 8,
                        "token_id": batch * 100 + i,
                        "routing_weight": 0.1 + (i % 10) * 0.1,
                        "layer_id": i % 4,
                        "timestamp": time.time()
                    }
                    for i in range(100)  # 100 events per batch
                ]
                
                for event in events:
                    enhanced_debugger.process_routing_event(event)
                
                # Periodic analysis
                if batch % 3 == 0:
                    enhanced_debugger.analyze_expert_utilization()
                    enhanced_debugger.get_performance_metrics()
            
            # 3. Final analysis
            final_stats = {
                "utilization": enhanced_debugger.analyze_expert_utilization(),
                "performance": enhanced_debugger.get_performance_metrics(),
                "routing_stats": enhanced_debugger.get_routing_stats()
            }
            
            # 4. End session
            enhanced_debugger.end_session()
            
            return final_stats
        
        session_results = benchmark(debugging_session)
        assert session_results is not None
        
        # Realistic session metrics
        benchmark.extra_info.update({
            "custom_metrics": {
                "total_events_processed": 1000,
                "analysis_calls": 4,  # 3 periodic + 1 final
                "session_duration_target_met": benchmark.stats.mean < 5.0,  # < 5 seconds
                "realistic_performance": benchmark.stats.mean < 2.0  # < 2 seconds for good UX
            }
        })
    
    def test_high_frequency_streaming(self, benchmark, enhanced_debugger):
        """Benchmark high-frequency real-time streaming scenario."""
        
        def streaming_simulation():
            # Simulate high-frequency streaming like real-time inference
            total_events = 0
            batch_size = 50
            num_batches = 100
            
            for batch_idx in range(num_batches):
                # Generate batch of events
                events = [
                    {
                        "expert_id": (batch_idx * batch_size + i) % 8,
                        "token_id": batch_idx * batch_size + i,
                        "routing_weight": 0.05 + ((batch_idx + i) % 20) * 0.045,
                        "layer_id": (batch_idx + i) % 6,
                        "timestamp": time.time(),
                        "sequence_id": batch_idx
                    }
                    for i in range(batch_size)
                ]
                
                # Process events quickly
                for event in events:
                    enhanced_debugger.process_routing_event(event)
                    total_events += 1
                
                # Real-time analysis every 10 batches
                if batch_idx % 10 == 0:
                    enhanced_debugger.get_routing_stats()
            
            return total_events
        
        total_processed = benchmark(streaming_simulation)
        assert total_processed == 5000
        
        # Streaming performance metrics
        events_per_second = total_processed / benchmark.stats.mean
        benchmark.extra_info.update({
            "custom_metrics": {
                "streaming_events_per_second": events_per_second,
                "high_throughput_achieved": events_per_second >= 2000,
                "real_time_capable": benchmark.stats.mean < 3.0,  # < 3 seconds for 5000 events
                "batch_processing_efficiency": total_processed / (100 * 50)  # Should be 1.0
            }
        })


# Benchmark configuration
pytest_benchmark_options = {
    "min_rounds": 5,
    "max_time": 10.0,
    "warmup": True,
    "warmup_iterations": 2,
    "timer": time.perf_counter,
    "sort": "mean"
}


if __name__ == "__main__":
    # Run specific benchmark
    pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-json=benchmark-results.json",
        "--benchmark-columns=min,max,mean,stddev,rounds,iterations",
        "-v"
    ])