"""Performance profiling engine for MoE models."""

import torch
import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np

from .models import PerformanceProfile


class MoEProfiler:
    """Performance profiling for MoE models with memory and compute tracking."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.is_profiling = False
        self.profiles = deque(maxlen=1000)
        
        # Timing tracking
        self.start_time = None
        self.layer_times = defaultdict(list)
        self.expert_times = defaultdict(list)
        self.routing_times = defaultdict(list)
        
        # Memory tracking
        self.memory_snapshots = []
        self.peak_memory = 0
        
        # Cache tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance counters
        self.token_count = 0
        self.inference_count = 0
        
        # Threading
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
    
    def start_profiling(self):
        """Start performance profiling."""
        with self.lock:
            if self.is_profiling:
                return
            
            self.is_profiling = True
            self.start_time = time.perf_counter()
            self.stop_monitoring.clear()
            
            # Start memory monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_memory)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # Reset CUDA memory stats if available
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
    
    def stop_profiling(self):
        """Stop performance profiling."""
        with self.lock:
            if not self.is_profiling:
                return
            
            self.is_profiling = False
            self.stop_monitoring.set()
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            
            # Create final profile
            self._create_profile()
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while not self.stop_monitoring.wait(0.1):  # Check every 100ms
            try:
                # System memory
                system_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
                
                # GPU memory if available
                gpu_memory = 0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    self.peak_memory = max(self.peak_memory, gpu_memory)
                
                snapshot = {
                    "timestamp": time.time(),
                    "system_memory_mb": system_memory,
                    "gpu_memory_mb": gpu_memory
                }
                
                with self.lock:
                    self.memory_snapshots.append(snapshot)
                    
                    # Keep only recent snapshots
                    if len(self.memory_snapshots) > 1000:
                        self.memory_snapshots = self.memory_snapshots[-500:]
                        
            except Exception as e:
                # Continue monitoring even if snapshot fails
                pass
    
    @contextmanager
    def profile_inference(self):
        """Context manager for profiling a single inference."""
        start_time = time.perf_counter()
        start_memory = self._get_current_memory()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_current_memory()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            memory_delta = end_memory - start_memory
            
            with self.lock:
                self.inference_count += 1
                # Store timing for later analysis
                self.layer_times["total_inference"].append(inference_time)
    
    @contextmanager
    def profile_layer(self, layer_name: str):
        """Context manager for profiling a model layer."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            layer_time = (end_time - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.layer_times[layer_name].append(layer_time)
    
    @contextmanager
    def profile_expert(self, expert_id: int, layer_idx: int):
        """Context manager for profiling expert computation."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            expert_time = (end_time - start_time) * 1000  # Convert to ms
            
            expert_key = f"layer_{layer_idx}_expert_{expert_id}"
            with self.lock:
                self.expert_times[expert_key].append(expert_time)
    
    @contextmanager
    def profile_routing(self, layer_idx: int):
        """Context manager for profiling routing computation."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            routing_time = (end_time - start_time) * 1000  # Convert to ms
            
            routing_key = f"layer_{layer_idx}_routing"
            with self.lock:
                self.routing_times[routing_key].append(routing_time)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self.lock:
            self.cache_misses += 1
    
    def record_tokens_processed(self, count: int):
        """Record number of tokens processed."""
        with self.lock:
            self.token_count += count
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def _create_profile(self) -> PerformanceProfile:
        """Create a performance profile from collected data."""
        end_time = time.perf_counter()
        total_time = (end_time - (self.start_time or end_time)) * 1000  # Convert to ms
        
        # Calculate routing overhead
        total_routing_time = sum(
            sum(times) for times in self.routing_times.values()
        )
        
        # Calculate expert compute times
        expert_compute_times = {}
        for expert_key, times in self.expert_times.items():
            if times:
                expert_compute_times[expert_key] = sum(times)
        
        # Calculate cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        
        # Calculate token throughput
        token_throughput = self.token_count / (total_time / 1000) if total_time > 0 else 0.0
        
        profile = PerformanceProfile(
            total_inference_time_ms=total_time,
            routing_overhead_ms=total_routing_time,
            expert_compute_times=expert_compute_times,
            memory_peak_mb=self.peak_memory,
            cache_hit_rate=cache_hit_rate,
            token_throughput=token_throughput
        )
        
        self.profiles.append(profile)
        return profile
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            current_memory = self._get_current_memory()
            
            # Calculate averages
            avg_layer_times = {}
            for layer, times in self.layer_times.items():
                if times:
                    avg_layer_times[layer] = np.mean(times)
            
            avg_expert_times = {}
            for expert, times in self.expert_times.items():
                if times:
                    avg_expert_times[expert] = np.mean(times)
            
            total_cache_ops = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
            
            return {
                "is_profiling": self.is_profiling,
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.peak_memory,
                "cache_hit_rate": cache_hit_rate,
                "total_cache_operations": total_cache_ops,
                "tokens_processed": self.token_count,
                "inference_count": self.inference_count,
                "avg_layer_times_ms": avg_layer_times,
                "avg_expert_times_ms": avg_expert_times,
                "memory_history": self.memory_snapshots[-100:] if self.memory_snapshots else []
            }
    
    def get_profiles(self) -> List[PerformanceProfile]:
        """Get all collected performance profiles."""
        return list(self.profiles)
    
    def get_latest_profile(self) -> Optional[PerformanceProfile]:
        """Get the most recent performance profile."""
        return self.profiles[-1] if self.profiles else None
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks."""
        bottlenecks = []
        
        with self.lock:
            # Analyze layer timing
            total_layer_time = 0
            layer_percentages = {}
            
            for layer, times in self.layer_times.items():
                if times:
                    avg_time = np.mean(times)
                    total_layer_time += avg_time
                    layer_percentages[layer] = avg_time
            
            # Identify slow layers
            if total_layer_time > 0:
                for layer, time_ms in layer_percentages.items():
                    percentage = (time_ms / total_layer_time) * 100
                    if percentage > 30:  # Layer taking >30% of total time
                        bottlenecks.append({
                            "type": "slow_layer",
                            "component": layer,
                            "time_ms": time_ms,
                            "percentage": percentage,
                            "severity": "high" if percentage > 50 else "medium"
                        })
            
            # Analyze expert timing imbalance
            expert_times_by_layer = defaultdict(list)
            for expert_key, times in self.expert_times.items():
                if times and "layer_" in expert_key:
                    layer_part = expert_key.split("_expert_")[0]
                    avg_time = np.mean(times)
                    expert_times_by_layer[layer_part].append(avg_time)
            
            for layer, expert_times in expert_times_by_layer.items():
                if len(expert_times) > 1:
                    coeff_var = np.std(expert_times) / np.mean(expert_times)
                    if coeff_var > 0.5:  # High variation in expert times
                        bottlenecks.append({
                            "type": "expert_imbalance",
                            "component": layer,
                            "coefficient_of_variation": coeff_var,
                            "expert_times": expert_times,
                            "severity": "medium"
                        })
            
            # Memory analysis
            if self.peak_memory > 8000:  # >8GB
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "component": "memory",
                    "peak_memory_mb": self.peak_memory,
                    "severity": "high" if self.peak_memory > 16000 else "medium"
                })
            
            # Cache analysis
            total_cache_ops = self.cache_hits + self.cache_misses
            if total_cache_ops > 0:
                cache_hit_rate = self.cache_hits / total_cache_ops
                if cache_hit_rate < 0.5:  # <50% hit rate
                    bottlenecks.append({
                        "type": "poor_cache_performance",
                        "component": "cache",
                        "hit_rate": cache_hit_rate,
                        "total_operations": total_cache_ops,
                        "severity": "medium"
                    })
        
        return bottlenecks
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions."""
        suggestions = []
        bottlenecks = self.analyze_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_layer":
                suggestions.append(f"Consider optimizing {bottleneck['component']} which takes {bottleneck['percentage']:.1f}% of compute time")
            
            elif bottleneck["type"] == "expert_imbalance":
                suggestions.append(f"Expert computation times are imbalanced in {bottleneck['component']} (CV: {bottleneck['coefficient_of_variation']:.2f})")
            
            elif bottleneck["type"] == "high_memory_usage":
                suggestions.append(f"High memory usage detected ({bottleneck['peak_memory_mb']:.0f}MB) - consider model parallelism or gradient checkpointing")
            
            elif bottleneck["type"] == "poor_cache_performance":
                suggestions.append(f"Low cache hit rate ({bottleneck['hit_rate']:.1%}) - consider increasing cache size or improving locality")
        
        if not suggestions:
            suggestions.append("Performance appears optimal based on current analysis")
        
        return suggestions
    
    def export_profile_data(self) -> Dict[str, Any]:
        """Export all profiling data for analysis."""
        with self.lock:
            return {
                "layer_times": dict(self.layer_times),
                "expert_times": dict(self.expert_times),
                "routing_times": dict(self.routing_times),
                "memory_snapshots": self.memory_snapshots,
                "cache_stats": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                },
                "counters": {
                    "tokens_processed": self.token_count,
                    "inference_count": self.inference_count,
                    "peak_memory_mb": self.peak_memory
                },
                "profiles": [
                    {
                        "total_inference_time_ms": p.total_inference_time_ms,
                        "routing_overhead_ms": p.routing_overhead_ms,
                        "expert_compute_times": p.expert_compute_times,
                        "memory_peak_mb": p.memory_peak_mb,
                        "cache_hit_rate": p.cache_hit_rate,
                        "token_throughput": p.token_throughput,
                        "timestamp": p.timestamp.isoformat()
                    }
                    for p in self.profiles
                ]
            }
    
    def clear_data(self):
        """Clear all profiling data."""
        with self.lock:
            self.layer_times.clear()
            self.expert_times.clear()
            self.routing_times.clear()
            self.memory_snapshots.clear()
            self.profiles.clear()
            
            self.cache_hits = 0
            self.cache_misses = 0
            self.token_count = 0
            self.inference_count = 0
            self.peak_memory = 0
    
    def __enter__(self):
        """Support for context manager."""
        self.start_profiling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager."""
        self.stop_profiling()