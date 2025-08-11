"""Advanced optimization engine for maximum performance and scalability."""

import asyncio
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from queue import PriorityQueue, Queue
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTask:
    """Task for optimization processing."""
    priority: int
    timestamp: float
    task_id: str
    task_type: str
    payload: Any
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        # Higher priority tasks are processed first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


@dataclass
class OptimizationMetrics:
    """Metrics for optimization engine performance."""
    tasks_processed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    queue_depth: int = 0
    worker_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    last_updated: float = field(default_factory=time.time)


class AdvancedOptimizationEngine:
    """High-performance optimization engine with intelligent scheduling."""
    
    def __init__(self, 
                 max_workers: int = 8,
                 max_queue_size: int = 10000,
                 enable_multiprocessing: bool = False):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_multiprocessing = enable_multiprocessing
        
        # Task processing
        self.task_queue = PriorityQueue(maxsize=max_queue_size)
        self.processing_workers: List[threading.Thread] = []
        self.is_running = False
        
        # Executor pools
        if enable_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        
        # Performance monitoring
        self.metrics = OptimizationMetrics()
        self.processing_times = deque(maxlen=1000)  # Rolling window
        self.last_throughput_check = time.time()
        self.throughput_counter = 0
        
        # Adaptive optimization
        self.adaptive_config = {
            'batch_size': 10,
            'batch_timeout': 1.0,
            'priority_boost_threshold': 5.0,  # seconds
            'load_balance_factor': 0.8
        }
        
        # Resource monitoring
        self._lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        with self._lock:
            self.task_handlers[task_type] = handler
            logger.info(f"Registered handler for task type: {task_type}")
    
    def submit_task(self, 
                   task_type: str,
                   payload: Any,
                   priority: int = 5,
                   callback: Optional[Callable] = None) -> str:
        """Submit a task for processing."""
        task_id = f"{task_type}_{int(time.time()*1000000)}"
        
        task = OptimizationTask(
            priority=priority,
            timestamp=time.time(),
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            callback=callback
        )
        
        try:
            self.task_queue.put(task, timeout=1.0)
            logger.debug(f"Submitted task {task_id} with priority {priority}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to submit task {task_id}: {e}")
            raise
    
    def submit_batch(self, tasks: List[Tuple[str, Any, int]]) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        for task_type, payload, priority in tasks:
            task_id = self.submit_task(task_type, payload, priority)
            task_ids.append(task_id)
        
        logger.debug(f"Submitted batch of {len(tasks)} tasks")
        return task_ids
    
    def start(self):
        """Start the optimization engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"OptimizationWorker-{i}",
                daemon=True
            )
            worker.start()
            self.processing_workers.append(worker)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="OptimizationMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Optimization engine started with {self.max_workers} workers")
    
    def stop(self, timeout: float = 10.0):
        """Stop the optimization engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.processing_workers:
            remaining_time = max(0, timeout - (time.time() - start_time))
            if remaining_time > 0:
                worker.join(timeout=remaining_time)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Optimization engine stopped")
    
    def _worker_loop(self):
        """Main worker processing loop."""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Try to get a task
                try:
                    task = self.task_queue.get(timeout=0.1)
                    batch.append(task)
                except:
                    task = None
                
                # Process batch if conditions are met
                should_process_batch = (
                    len(batch) >= self.adaptive_config['batch_size'] or
                    (batch and time.time() - last_batch_time >= self.adaptive_config['batch_timeout'])
                )
                
                if should_process_batch and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                
                # Priority boost for old tasks
                self._boost_old_task_priorities()
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(0.1)
        
        # Process remaining batch
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[OptimizationTask]):
        """Process a batch of tasks."""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Group tasks by type for efficient processing
            tasks_by_type = defaultdict(list)
            for task in batch:
                tasks_by_type[task.task_type].append(task)
            
            # Process each task type
            for task_type, tasks in tasks_by_type.items():
                if task_type in self.task_handlers:
                    try:
                        # Batch process if handler supports it
                        handler = self.task_handlers[task_type]
                        if hasattr(handler, 'process_batch'):
                            results = handler.process_batch([task.payload for task in tasks])
                            
                            # Call individual callbacks
                            for task, result in zip(tasks, results):
                                if task.callback:
                                    try:
                                        task.callback(result)
                                    except Exception as e:
                                        logger.error(f"Callback failed for task {task.task_id}: {e}")
                        else:
                            # Process individually
                            for task in tasks:
                                try:
                                    result = handler(task.payload)
                                    if task.callback:
                                        task.callback(result)
                                    
                                    self.metrics.tasks_processed += 1
                                    
                                except Exception as e:
                                    logger.error(f"Task {task.task_id} failed: {e}")
                                    self.metrics.tasks_failed += 1
                    
                    except Exception as e:
                        logger.error(f"Batch processing failed for {task_type}: {e}")
                        self.metrics.tasks_failed += len(tasks)
                else:
                    logger.warning(f"No handler for task type: {task_type}")
                    self.metrics.tasks_failed += len(tasks)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update average processing time
            if self.processing_times:
                self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.metrics.tasks_failed += len(batch)
    
    def _boost_old_task_priorities(self):
        """Boost priority of tasks that have been waiting too long."""
        current_time = time.time()
        threshold = self.adaptive_config['priority_boost_threshold']
        
        # This is a simplified implementation
        # In a real system, you'd need a more sophisticated approach
        # to avoid modifying the queue during iteration
        pass
    
    def _monitor_loop(self):
        """Monitor performance and adjust configuration."""
        while self.is_running:
            try:
                # Update throughput metrics
                current_time = time.time()
                time_diff = current_time - self.last_throughput_check
                
                if time_diff >= 1.0:  # Update every second
                    self.metrics.throughput_per_second = self.throughput_counter / time_diff
                    self.throughput_counter = 0
                    self.last_throughput_check = current_time
                    
                    # Update queue depth
                    self.metrics.queue_depth = self.task_queue.qsize()
                    
                    # Calculate worker utilization
                    active_workers = sum(1 for worker in self.processing_workers if worker.is_alive())
                    self.metrics.worker_utilization = active_workers / self.max_workers
                    
                    # Update memory usage
                    try:
                        import psutil
                        process = psutil.Process()
                        self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                    except:
                        pass
                    
                    self.metrics.last_updated = current_time
                
                # Adaptive configuration adjustment
                self._adjust_adaptive_config()
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(1.0)
    
    def _adjust_adaptive_config(self):
        """Adjust configuration based on performance metrics."""
        # Adjust batch size based on queue depth
        if self.metrics.queue_depth > 100:
            self.adaptive_config['batch_size'] = min(20, self.adaptive_config['batch_size'] + 1)
        elif self.metrics.queue_depth < 10:
            self.adaptive_config['batch_size'] = max(1, self.adaptive_config['batch_size'] - 1)
        
        # Adjust batch timeout based on throughput
        if self.metrics.throughput_per_second < 10:
            self.adaptive_config['batch_timeout'] = max(0.1, self.adaptive_config['batch_timeout'] - 0.1)
        elif self.metrics.throughput_per_second > 100:
            self.adaptive_config['batch_timeout'] = min(2.0, self.adaptive_config['batch_timeout'] + 0.1)
    
    def get_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        with self._lock:
            return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        metrics = self.get_metrics()
        
        return {
            "is_running": self.is_running,
            "workers": {
                "max_workers": self.max_workers,
                "active_workers": sum(1 for w in self.processing_workers if w.is_alive()),
                "worker_utilization": metrics.worker_utilization
            },
            "queue": {
                "size": metrics.queue_depth,
                "max_size": self.max_queue_size,
                "utilization": metrics.queue_depth / self.max_queue_size
            },
            "performance": {
                "tasks_processed": metrics.tasks_processed,
                "tasks_failed": metrics.tasks_failed,
                "success_rate": (metrics.tasks_processed / max(1, metrics.tasks_processed + metrics.tasks_failed)) * 100,
                "average_processing_time": metrics.average_processing_time,
                "throughput_per_second": metrics.throughput_per_second
            },
            "adaptive_config": self.adaptive_config.copy(),
            "memory_usage_mb": metrics.memory_usage_mb,
            "last_updated": metrics.last_updated
        }


# Task handler examples
class BatchRoutingEventProcessor:
    """Example batch processor for routing events."""
    
    def process_batch(self, events: List[Any]) -> List[Any]:
        """Process a batch of routing events efficiently."""
        results = []
        
        # Simulate batch processing
        for event in events:
            # Process routing event
            result = {
                "event_id": event.get("id", "unknown"),
                "processed": True,
                "timestamp": time.time()
            }
            results.append(result)
        
        return results


class ModelAnalysisProcessor:
    """Example processor for model analysis tasks."""
    
    def __call__(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process model analysis request."""
        # Simulate analysis
        time.sleep(0.01)  # Simulate processing time
        
        return {
            "analysis_id": analysis_request.get("id", "unknown"),
            "model_metrics": {
                "utilization": 0.75,
                "efficiency": 0.82,
                "recommendations": ["Optimize expert routing", "Increase batch size"]
            },
            "completed_at": time.time()
        }


# Global optimization engine instance
_global_optimization_engine: Optional[AdvancedOptimizationEngine] = None


def get_optimization_engine() -> AdvancedOptimizationEngine:
    """Get global optimization engine instance."""
    global _global_optimization_engine
    if _global_optimization_engine is None:
        _global_optimization_engine = AdvancedOptimizationEngine()
        
        # Register default handlers
        _global_optimization_engine.register_task_handler(
            "routing_events", 
            BatchRoutingEventProcessor()
        )
        _global_optimization_engine.register_task_handler(
            "model_analysis",
            ModelAnalysisProcessor()
        )
    
    return _global_optimization_engine