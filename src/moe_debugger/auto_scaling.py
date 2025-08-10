"""
Auto-scaling and load balancing for MoE Debugger.
Implements intelligent scaling based on load patterns and resource usage.
"""

import time
import threading
import psutil
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import os

from .logging_config import get_logger
from .monitoring import HealthMonitor

logger = get_logger(__name__)


@dataclass
class LoadMetrics:
    """System load and performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    requests_per_second: float
    active_connections: int
    queue_length: int
    response_time_ms: float


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 2
    max_workers: int = 16
    cpu_threshold_high: float = 75.0
    cpu_threshold_low: float = 30.0
    memory_threshold_high: float = 80.0
    memory_threshold_low: float = 40.0
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    metrics_window: int = 60  # seconds
    min_samples: int = 3


class AutoScaler:
    """Intelligent auto-scaling system for MoE debugging workloads."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.metrics_history: deque = deque(maxlen=self.config.metrics_window)
        self.current_workers = self.config.min_workers
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="moe_debugger_"
        )
        
        # Process pool for heavy computation
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(self.config.max_workers // 2, os.cpu_count() or 2)
        )
        
        # Monitoring and control
        self._stop_monitoring = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._lock = threading.Lock()
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.request_count = 0
        self.connection_count = 0
        
        logger.info(f"AutoScaler initialized with {self.current_workers} workers")
    
    def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        if not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop the auto-scaling monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop for auto-scaling decisions."""
        while not self._stop_monitoring.wait(10.0):  # Check every 10 seconds
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                if len(self.metrics_history) >= self.config.min_samples:
                    self._evaluate_scaling()
                    
            except Exception as e:
                logger.error(f"Error in auto-scaling monitor: {e}")
    
    def _collect_metrics(self) -> LoadMetrics:
        """Collect current system and application metrics."""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Application metrics
        with self._lock:
            # Calculate requests per second
            recent_requests = [
                t for t in self.request_times 
                if current_time - t < 60.0  # Last minute
            ]
            rps = len(recent_requests) / 60.0
            
            # Average response time
            if self.request_times:
                avg_response_time = statistics.mean([
                    100.0  # Mock response time for demo
                    for _ in self.request_times[-10:]  # Last 10 requests
                ])
            else:
                avg_response_time = 0.0
        
        return LoadMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            requests_per_second=rps,
            active_connections=self.connection_count,
            queue_length=self.thread_pool._work_queue.qsize(),
            response_time_ms=avg_response_time
        )
    
    def _evaluate_scaling(self):
        """Evaluate whether scaling is needed based on metrics."""
        recent_metrics = list(self.metrics_history)[-self.config.min_samples:]
        
        # Calculate averages
        avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
        avg_queue_length = statistics.mean([m.queue_length for m in recent_metrics])
        avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
        
        current_time = time.time()
        
        # Scale up conditions
        should_scale_up = (
            (avg_cpu > self.config.cpu_threshold_high or
             avg_memory > self.config.memory_threshold_high or
             avg_queue_length > self.current_workers * 2 or
             avg_response_time > 500.0) and
            self.current_workers < self.config.max_workers and
            (current_time - self.last_scale_up) > self.config.scale_up_cooldown
        )
        
        # Scale down conditions
        should_scale_down = (
            avg_cpu < self.config.cpu_threshold_low and
            avg_memory < self.config.memory_threshold_low and
            avg_queue_length < self.current_workers * 0.5 and
            avg_response_time < 200.0 and
            self.current_workers > self.config.min_workers and
            (current_time - self.last_scale_down) > self.config.scale_down_cooldown
        )
        
        if should_scale_up:
            self._scale_up()
        elif should_scale_down:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up the worker pools."""
        old_workers = self.current_workers
        self.current_workers = min(self.current_workers + 2, self.config.max_workers)
        
        if self.current_workers > old_workers:
            # Replace thread pool with larger one
            old_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix="moe_debugger_"
            )
            
            # Gracefully shutdown old pool
            threading.Thread(target=lambda: old_pool.shutdown(wait=True), daemon=True).start()
            
            self.last_scale_up = time.time()
            logger.info(f"Scaled up from {old_workers} to {self.current_workers} workers")
    
    def _scale_down(self):
        """Scale down the worker pools."""
        old_workers = self.current_workers
        self.current_workers = max(self.current_workers - 1, self.config.min_workers)
        
        if self.current_workers < old_workers:
            # Replace thread pool with smaller one
            old_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix="moe_debugger_"
            )
            
            # Gracefully shutdown old pool
            threading.Thread(target=lambda: old_pool.shutdown(wait=True), daemon=True).start()
            
            self.last_scale_down = time.time()
            logger.info(f"Scaled down from {old_workers} to {self.current_workers} workers")
    
    def record_request(self, response_time_ms: Optional[float] = None):
        """Record a completed request for metrics."""
        with self._lock:
            self.request_times.append(time.time())
            self.request_count += 1
    
    def increment_connections(self):
        """Increment active connection count."""
        with self._lock:
            self.connection_count += 1
    
    def decrement_connections(self):
        """Decrement active connection count."""
        with self._lock:
            self.connection_count = max(0, self.connection_count - 1)
    
    def submit_task(self, fn: Callable, *args, **kwargs):
        """Submit a task to the thread pool."""
        return self.thread_pool.submit(fn, *args, **kwargs)
    
    def submit_cpu_intensive_task(self, fn: Callable, *args, **kwargs):
        """Submit a CPU-intensive task to the process pool."""
        return self.process_pool.submit(fn, *args, **kwargs)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get current scaling statistics."""
        recent_metrics = list(self.metrics_history)[-5:] if self.metrics_history else []
        
        if recent_metrics:
            latest = recent_metrics[-1]
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
        else:
            latest = None
            avg_cpu = avg_memory = 0.0
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'active_connections': self.connection_count,
            'total_requests': self.request_count,
            'recent_rps': latest.requests_per_second if latest else 0.0,
            'current_cpu': latest.cpu_percent if latest else 0.0,
            'current_memory': latest.memory_percent if latest else 0.0,
            'avg_cpu_5min': avg_cpu,
            'avg_memory_5min': avg_memory,
            'queue_length': self.thread_pool._work_queue.qsize(),
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down
        }
    
    def shutdown(self):
        """Gracefully shutdown all pools."""
        self.stop_monitoring()
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("AutoScaler shutdown complete")


class LoadBalancer:
    """Simple round-robin load balancer for distributed processing."""
    
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.current_index = 0
        self.backend_stats = {backend: {'requests': 0, 'failures': 0} for backend in backends}
        self._lock = threading.Lock()
    
    def get_next_backend(self) -> str:
        """Get the next backend in round-robin fashion."""
        with self._lock:
            backend = self.backends[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.backends)
            self.backend_stats[backend]['requests'] += 1
            return backend
    
    def record_failure(self, backend: str):
        """Record a failure for a backend."""
        with self._lock:
            if backend in self.backend_stats:
                self.backend_stats[backend]['failures'] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get load balancer statistics."""
        with self._lock:
            return dict(self.backend_stats)


# Global auto-scaler instance
_global_autoscaler: Optional[AutoScaler] = None


def get_global_autoscaler(config: Optional[ScalingConfig] = None) -> AutoScaler:
    """Get or create the global auto-scaler instance."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = AutoScaler(config)
        _global_autoscaler.start_monitoring()
    return _global_autoscaler


def shutdown_global_autoscaler():
    """Shutdown the global auto-scaler."""
    global _global_autoscaler
    if _global_autoscaler:
        _global_autoscaler.shutdown()
        _global_autoscaler = None