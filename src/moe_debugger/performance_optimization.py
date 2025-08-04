"""Performance optimization and scaling features for MoE debugging."""

import time
import threading
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import queue
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
import gc

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .logging_config import get_logger, performance_timer
from .caching import get_global_cache, cached, CacheKey
from .models import RoutingEvent

logger = get_logger(__name__)


@dataclass
class ProcessingTask:
    """Task for async processing."""
    task_id: str
    operation: str
    data: Any
    priority: int = 0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class AsyncTaskProcessor:
    """High-performance async task processor with prioritization."""
    
    def __init__(self, max_workers: int = None, queue_size: int = 10000):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.task_queue = asyncio.PriorityQueue(maxsize=queue_size)
        self.result_cache = {}
        self.workers = []
        self.running = False
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'queue_size': 0
        }
        self._task_counter = 0
        
    async def start(self):
        """Start the task processor."""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started async task processor with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the task processor."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Stopped async task processor")
    
    async def submit_task(self, operation: str, data: Any, 
                         priority: int = 0) -> str:
        """Submit a task for processing."""
        task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        task = ProcessingTask(
            task_id=task_id,
            operation=operation,
            data=data,
            priority=priority
        )
        
        try:
            # Use negative priority for max-heap behavior (higher number = higher priority)
            await self.task_queue.put((-priority, task_id, task))
            self.stats['queue_size'] = self.task_queue.qsize()
            return task_id
        except asyncio.QueueFull:
            logger.warning("Task queue full, dropping task")
            raise
    
    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get result of a submitted task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.result_cache:
                result = self.result_cache.pop(task_id)
                if isinstance(result, Exception):
                    raise result
                return result
            
            await asyncio.sleep(0.01)  # Small delay
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine to process tasks."""
        while self.running:
            try:
                # Get task from queue
                priority, task_id, task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                start_time = time.time()
                
                # Process task
                try:
                    result = await self._process_task(task)
                    self.result_cache[task_id] = result
                    self.stats['tasks_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    self.result_cache[task_id] = e
                    self.stats['tasks_failed'] += 1
                
                # Update stats
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                self.stats['queue_size'] = self.task_queue.qsize()
                
                # Mark task as done
                self.task_queue.task_done()
            
            except asyncio.TimeoutError:
                continue  # No tasks available, continue waiting
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_task(self, task: ProcessingTask) -> Any:
        """Process a single task."""
        if task.operation == "analyze_routing_events":
            return await self._analyze_routing_events(task.data)
        elif task.operation == "compute_statistics":
            return await self._compute_statistics(task.data)
        elif task.operation == "detect_anomalies":
            return await self._detect_anomalies(task.data)
        else:
            raise ValueError(f"Unknown operation: {task.operation}")
    
    async def _analyze_routing_events(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Analyze routing events asynchronously."""
        # This would be run in a thread pool for CPU-intensive work
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_analyze_routing_events, events)
    
    def _sync_analyze_routing_events(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Synchronous routing analysis."""
        if not events:
            return {}
        
        # Fast analysis using vectorized operations if numpy available
        if NUMPY_AVAILABLE:
            return self._vectorized_routing_analysis(events)
        else:
            return self._standard_routing_analysis(events)
    
    def _vectorized_routing_analysis(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Vectorized analysis using numpy."""
        # Extract data into arrays
        weights_array = np.array([event.expert_weights for event in events])
        confidence_array = np.array([event.routing_confidence for event in events])
        
        # Compute statistics efficiently
        return {
            'total_events': len(events),
            'mean_confidence': np.mean(confidence_array),
            'std_confidence': np.std(confidence_array),
            'expert_usage': np.mean(weights_array, axis=0).tolist(),
            'routing_entropy': self._compute_entropy_vectorized(weights_array)
        }
    
    def _compute_entropy_vectorized(self, weights_array: np.ndarray) -> float:
        """Compute routing entropy using vectorized operations."""
        # Softmax normalization
        exp_weights = np.exp(weights_array - np.max(weights_array, axis=1, keepdims=True))
        probs = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)
        
        # Entropy calculation
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return np.mean(entropy)
    
    def _standard_routing_analysis(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Standard analysis without numpy."""
        confidence_values = [event.routing_confidence for event in events]
        
        return {
            'total_events': len(events),
            'mean_confidence': statistics.mean(confidence_values),
            'std_confidence': statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0,
        }
    
    async def _compute_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics asynchronously."""
        # Placeholder for statistical computations
        await asyncio.sleep(0.001)  # Simulate async work
        return {"computed": True, "timestamp": time.time()}
    
    async def _detect_anomalies(self, data: List[Any]) -> List[Dict[str, Any]]:
        """Detect anomalies asynchronously."""
        # Placeholder for anomaly detection
        await asyncio.sleep(0.001)  # Simulate async work
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self.stats.copy()
        
        if stats['tasks_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['tasks_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        stats['queue_utilization'] = self.task_queue.qsize() / self.task_queue.maxsize
        stats['active_workers'] = len([w for w in self.workers if not w.done()])
        
        return stats


class BatchProcessor:
    """Efficient batch processing for large datasets."""
    
    def __init__(self, batch_size: int = 1000, max_workers: int = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.cache = get_global_cache()
        
    @performance_timer("batch_processor", "process_routing_events")
    def process_routing_events_batch(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Process routing events in batches for better performance."""
        if not events:
            return {}
        
        # Check cache first
        cache_key = CacheKey(
            "routing_analysis", 
            "batch_process",
            {"event_count": len(events), "batch_size": self.batch_size}
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for routing events batch ({len(events)} events)")
            return cached_result
        
        # Split into batches
        batches = [
            events[i:i + self.batch_size]
            for i in range(0, len(events), self.batch_size)
        ]
        
        logger.info(f"Processing {len(events)} events in {len(batches)} batches")
        
        # Process batches in parallel
        if len(batches) > 1 and self.max_workers > 1:
            results = self._process_batches_parallel(batches)
        else:
            results = [self._process_single_batch(batch) for batch in batches]
        
        # Merge results
        merged_result = self._merge_batch_results(results)
        
        # Cache result
        self.cache.set(cache_key, merged_result, ttl=300)  # 5 minutes
        
        return merged_result
    
    def _process_batches_parallel(self, batches: List[List[RoutingEvent]]) -> List[Dict[str, Any]]:
        """Process batches in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_single_batch, batch): i
                for i, batch in enumerate(batches)
            }
            
            results = [None] * len(batches)
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results[batch_idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch {batch_idx} processing failed: {e}")
                    results[batch_idx] = {"error": str(e)}
        
        return results
    
    def _process_single_batch(self, events: List[RoutingEvent]) -> Dict[str, Any]:
        """Process a single batch of events."""
        if not events:
            return {}
        
        # Extract metrics efficiently
        confidence_values = []
        expert_selections = defaultdict(int)
        layer_counts = defaultdict(int)
        
        for event in events:
            confidence_values.append(event.routing_confidence)
            layer_counts[event.layer_idx] += 1
            
            for expert_id in event.selected_experts:
                expert_selections[expert_id] += 1
        
        # Compute statistics
        result = {
            'event_count': len(events),
            'confidence_stats': {
                'mean': statistics.mean(confidence_values),
                'median': statistics.median(confidence_values),
                'std': statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0,
                'min': min(confidence_values),
                'max': max(confidence_values)
            },
            'expert_usage': dict(expert_selections),
            'layer_distribution': dict(layer_counts)
        }
        
        return result
    
    def _merge_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple batches."""
        if not results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {"error": "All batches failed"}
        
        # Merge statistics
        total_events = sum(r['event_count'] for r in valid_results)
        
        # Merge confidence statistics (weighted average)
        confidence_means = []
        confidence_stds = []
        confidence_mins = []
        confidence_maxs = []
        
        for result in valid_results:
            count = result['event_count']
            stats = result['confidence_stats']
            
            confidence_means.extend([stats['mean']] * count)
            confidence_stds.append(stats['std'])
            confidence_mins.append(stats['min'])
            confidence_maxs.append(stats['max'])
        
        # Merge expert usage
        merged_expert_usage = defaultdict(int)
        for result in valid_results:
            for expert_id, count in result['expert_usage'].items():
                merged_expert_usage[expert_id] += count
        
        # Merge layer distribution
        merged_layer_dist = defaultdict(int)
        for result in valid_results:
            for layer_id, count in result['layer_distribution'].items():
                merged_layer_dist[layer_id] += count
        
        return {
            'total_events': total_events,
            'batches_processed': len(valid_results),
            'confidence_stats': {
                'mean': statistics.mean(confidence_means),
                'std': statistics.mean(confidence_stds),
                'min': min(confidence_mins),
                'max': max(confidence_maxs)
            },
            'expert_usage': dict(merged_expert_usage),
            'layer_distribution': dict(merged_layer_dist)
        }


class ConnectionPool:
    """Connection pooling for database and external services."""
    
    def __init__(self, create_connection: Callable, max_connections: int = 10,
                 min_connections: int = 2, connection_timeout: float = 30.0):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        
        self._pool = queue.Queue(maxsize=max_connections)
        self._all_connections = set()
        self._lock = threading.Lock()
        self._stats = {
            'active_connections': 0,
            'total_created': 0,
            'total_borrowed': 0,
            'total_returned': 0,
            'connection_errors': 0
        }
        
        # Pre-create minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        for _ in range(self.min_connections):
            try:
                conn = self.create_connection()
                self._pool.put(conn)
                self._all_connections.add(conn)
                self._stats['total_created'] += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def get_connection(self, timeout: Optional[float] = None):
        """Get a connection from the pool."""
        timeout = timeout or self.connection_timeout
        
        try:
            # Try to get existing connection
            conn = self._pool.get(timeout=timeout)
            self._stats['total_borrowed'] += 1
            
            with self._lock:
                self._stats['active_connections'] += 1
            
            return ConnectionWrapper(self, conn)
            
        except queue.Empty:
            # Pool is empty, try to create new connection if under limit
            with self._lock:
                if len(self._all_connections) < self.max_connections:
                    try:
                        conn = self.create_connection()
                        self._all_connections.add(conn)
                        self._stats['total_created'] += 1
                        self._stats['total_borrowed'] += 1
                        self._stats['active_connections'] += 1
                        
                        return ConnectionWrapper(self, conn)
                        
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
                        self._stats['connection_errors'] += 1
                        raise
            
            raise TimeoutError("No connections available and max limit reached")
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
            self._stats['total_returned'] += 1
            
            with self._lock:
                self._stats['active_connections'] -= 1
                
        except queue.Full:
            # Pool is full, close excess connection
            self._close_connection(conn)
    
    def _close_connection(self, conn):
        """Close a connection."""
        try:
            if hasattr(conn, 'close'):
                conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            with self._lock:
                self._all_connections.discard(conn)
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            # Close connections in pool
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    self._close_connection(conn)
                except queue.Empty:
                    break
            
            # Close any remaining connections
            for conn in list(self._all_connections):
                self._close_connection(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'pool_size': self._pool.qsize(),
                'total_connections': len(self._all_connections),
                'max_connections': self.max_connections,
                'utilization': self._stats['active_connections'] / self.max_connections
            }


class ConnectionWrapper:
    """Wrapper for pooled connections with automatic return."""
    
    def __init__(self, pool: ConnectionPool, connection):
        self.pool = pool
        self.connection = connection
        self._returned = False
    
    def __enter__(self):
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.return_to_pool()
    
    def return_to_pool(self):
        """Return connection to pool."""
        if not self._returned:
            self.pool.return_connection(self.connection)
            self._returned = True
    
    def __getattr__(self, name):
        """Proxy all other attributes to the underlying connection."""
        return getattr(self.connection, name)


class PerformanceOptimizer:
    """Central performance optimization manager."""
    
    def __init__(self):
        self.async_processor = AsyncTaskProcessor()
        self.batch_processor = BatchProcessor()
        self.connection_pools = {}
        self.cache = get_global_cache()
        
        # Performance metrics
        self.metrics = {
            'request_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'cache_hit_rates': deque(maxlen=100)
        }
        
        # Auto-scaling parameters
        self.auto_scaling_config = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'response_time_threshold': 1.0,  # seconds
            'scale_up_cooldown': 300,  # 5 minutes
            'scale_down_cooldown': 600   # 10 minutes
        }
        
        self.last_scale_event = 0
    
    async def start(self):
        """Start all optimization components."""
        await self.async_processor.start()
        logger.info("Performance optimizer started")
    
    async def stop(self):
        """Stop all optimization components."""
        await self.async_processor.stop()
        
        # Close all connection pools
        for pool in self.connection_pools.values():
            pool.close_all()
        
        logger.info("Performance optimizer stopped")
    
    def add_connection_pool(self, name: str, create_func: Callable, 
                           max_connections: int = 10):
        """Add a connection pool."""
        self.connection_pools[name] = ConnectionPool(
            create_func, max_connections=max_connections
        )
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        return self.connection_pools.get(name)
    
    @performance_timer("optimizer", "process_large_dataset")
    async def process_large_dataset(self, data: List[Any], 
                                   operation: str) -> Dict[str, Any]:
        """Process large datasets efficiently."""
        if len(data) < 100:
            # Small dataset, process directly
            task_id = await self.async_processor.submit_task(operation, data, priority=1)
            return await self.async_processor.get_result(task_id)
        else:
            # Large dataset, use batch processing
            if operation == "analyze_routing_events":
                return self.batch_processor.process_routing_events_batch(data)
            else:
                # Fall back to async processing
                task_id = await self.async_processor.submit_task(operation, data, priority=2)
                return await self.async_processor.get_result(task_id)
    
    def record_performance_metric(self, metric_type: str, value: float):
        """Record a performance metric."""
        if metric_type in self.metrics:
            self.metrics[metric_type].append({
                'timestamp': time.time(),
                'value': value
            })
    
    def should_scale_up(self) -> bool:
        """Check if system should scale up."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_event < self.auto_scaling_config['scale_up_cooldown']:
            return False
        
        # Check CPU usage
        if self.metrics['cpu_usage']:
            recent_cpu = [m['value'] for m in list(self.metrics['cpu_usage'])[-5:]]
            avg_cpu = statistics.mean(recent_cpu)
            if avg_cpu > self.auto_scaling_config['cpu_threshold']:
                return True
        
        # Check response times
        if self.metrics['request_times']:
            recent_times = [m['value'] for m in list(self.metrics['request_times'])[-10:]]
            avg_time = statistics.mean(recent_times)
            if avg_time > self.auto_scaling_config['response_time_threshold']:
                return True
        
        return False
    
    def should_scale_down(self) -> bool:
        """Check if system should scale down."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_event < self.auto_scaling_config['scale_down_cooldown']:
            return False
        
        # Check if resources are underutilized
        if self.metrics['cpu_usage'] and len(self.metrics['cpu_usage']) >= 10:
            recent_cpu = [m['value'] for m in list(self.metrics['cpu_usage'])[-10:]]
            avg_cpu = statistics.mean(recent_cpu)
            
            if avg_cpu < 30.0:  # Low CPU usage  
                return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'async_processor': self.async_processor.get_stats(),
            'cache': self.cache.get_stats(),
            'connection_pools': {
                name: pool.get_stats()
                for name, pool in self.connection_pools.items()
            }
        }
        
        # Add metric summaries
        for metric_type, values in self.metrics.items():
            if values:
                recent_values = [v['value'] for v in list(values)[-10:]]
                stats[f'{metric_type}_summary'] = {
                    'mean': statistics.mean(recent_values),
                    'max': max(recent_values),
                    'min': min(recent_values),
                    'count': len(recent_values)
                }
        
        return stats
    
    def optimize_memory(self):
        """Perform memory optimization."""
        # Force garbage collection
        gc.collect()
        
        # Clear old cache entries
        self.cache.clear_by_tag("temporary")
        
        # Log memory optimization
        logger.info("Performed memory optimization")
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions."""
        suggestions = []
        stats = self.get_performance_stats()
        
        # Cache hit rate suggestions
        cache_stats = stats.get('cache', {})
        hit_rate = cache_stats.get('hit_rate', 0)
        
        if hit_rate < 0.5:
            suggestions.append("Consider increasing cache size or TTL values")
        
        # Async processor suggestions
        processor_stats = stats.get('async_processor', {})
        queue_util = processor_stats.get('queue_utilization', 0)
        
        if queue_util > 0.8:
            suggestions.append("Consider increasing async processor workers")
        
        # Connection pool suggestions
        for pool_name, pool_stats in stats.get('connection_pools', {}).items():
            utilization = pool_stats.get('utilization', 0)
            if utilization > 0.9:
                suggestions.append(f"Consider increasing {pool_name} connection pool size")
        
        if not suggestions:
            suggestions.append("System performance appears optimal")
        
        return suggestions


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_global_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    
    with _optimizer_lock:
        if _global_optimizer is None:
            _global_optimizer = PerformanceOptimizer()
        return _global_optimizer