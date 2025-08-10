"""
Distributed processing capabilities for MoE Debugger.
Enables horizontal scaling across multiple nodes and GPUs.
"""

import time
import hashlib
import json
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .logging_config import get_logger
from .models import RoutingEvent
from .auto_scaling import get_global_autoscaler

logger = get_logger(__name__)


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system."""
    node_id: str
    hostname: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    load_score: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True


@dataclass
class DistributedTask:
    """A task to be processed in the distributed system."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskQueue:
    """Distributed task queue using Redis or in-memory fallback."""
    
    def __init__(self, redis_url: Optional[str] = None, queue_name: str = "moe_debugger_tasks"):
        self.queue_name = queue_name
        self.redis_client = None
        self.local_queue = queue.PriorityQueue()
        self._use_redis = False
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self._use_redis = True
                logger.info(f"Connected to Redis for distributed task queue: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using local queue.")
        
        if not self._use_redis:
            logger.info("Using in-memory task queue (not distributed)")
    
    def enqueue(self, task: DistributedTask) -> bool:
        """Add a task to the queue."""
        try:
            if self._use_redis:
                # Use priority score (higher priority = lower score for Redis sorted sets)
                score = -task.priority + (task.created_at / 1000000)  # Add timestamp for ordering
                task_data = {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'payload': json.dumps(task.payload),
                    'priority': task.priority,
                    'created_at': task.created_at,
                    'status': task.status
                }
                self.redis_client.zadd(self.queue_name, {json.dumps(task_data): score})
            else:
                # Use negative priority for max-heap behavior
                self.local_queue.put((-task.priority, task.created_at, task))
            
            logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    def dequeue(self, timeout: float = 1.0) -> Optional[DistributedTask]:
        """Get the next task from the queue."""
        try:
            if self._use_redis:
                # Get highest priority task (lowest score)
                result = self.redis_client.zrange(self.queue_name, 0, 0, withscores=True)
                if result:
                    task_json, score = result[0]
                    # Remove from queue
                    self.redis_client.zrem(self.queue_name, task_json)
                    
                    task_data = json.loads(task_json)
                    task = DistributedTask(
                        task_id=task_data['task_id'],
                        task_type=task_data['task_type'],
                        payload=json.loads(task_data['payload']),
                        priority=task_data['priority'],
                        created_at=task_data['created_at'],
                        status=task_data['status']
                    )
                    return task
            else:
                try:
                    priority, created_at, task = self.local_queue.get(timeout=timeout)
                    return task
                except queue.Empty:
                    pass
            
            return None
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        try:
            if self._use_redis:
                return self.redis_client.zcard(self.queue_name)
            else:
                return self.local_queue.qsize()
        except Exception:
            return 0


class DistributedCoordinator:
    """Coordinates distributed processing across multiple nodes."""
    
    def __init__(self, node_id: Optional[str] = None, redis_url: Optional[str] = None):
        self.node_id = node_id or f"node_{int(time.time() * 1000) % 10000}"
        self.task_queue = TaskQueue(redis_url)
        self.nodes: Dict[str, ProcessingNode] = {}
        self.task_processors: Dict[str, Callable] = {}
        
        # Processing control
        self._stop_processing = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        self._is_running = False
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'node_count': 0
        }
        
        logger.info(f"Initialized distributed coordinator for node {self.node_id}")
    
    def register_processor(self, task_type: str, processor_func: Callable):
        """Register a processor function for a specific task type."""
        self.task_processors[task_type] = processor_func
        logger.info(f"Registered processor for task type: {task_type}")
    
    def submit_task(self, task_type: str, payload: Dict[str, Any], 
                   priority: int = 0) -> str:
        """Submit a task for distributed processing."""
        task_id = hashlib.md5(f"{task_type}_{time.time()}_{self.node_id}".encode()).hexdigest()
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        if self.task_queue.enqueue(task):
            logger.info(f"Submitted task {task_id} of type {task_type}")
            return task_id
        else:
            raise RuntimeError(f"Failed to submit task {task_id}")
    
    def start_processing(self, num_workers: int = None):
        """Start distributed task processing."""
        if self._is_running:
            logger.warning("Processing already running")
            return
        
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)
        
        self._is_running = True
        self._stop_processing.clear()
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"DistributedWorker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
        
        logger.info(f"Started distributed processing with {num_workers} workers")
    
    def stop_processing(self):
        """Stop distributed task processing."""
        if not self._is_running:
            return
        
        self._stop_processing.set()
        self._is_running = False
        
        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5.0)
        
        self._worker_threads.clear()
        logger.info("Stopped distributed processing")
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        autoscaler = get_global_autoscaler()
        
        while not self._stop_processing.is_set():
            try:
                task = self.task_queue.dequeue(timeout=1.0)
                if task is None:
                    continue
                
                # Record task start
                start_time = time.time()
                autoscaler.record_request()
                
                # Process the task
                if task.task_type in self.task_processors:
                    try:
                        task.status = "processing"
                        task.assigned_node = self.node_id
                        
                        processor = self.task_processors[task.task_type]
                        result = processor(task.payload)
                        
                        task.status = "completed"
                        task.result = result
                        
                        processing_time = time.time() - start_time
                        self.stats['tasks_processed'] += 1
                        
                        # Update average processing time
                        if self.stats['average_processing_time'] == 0:
                            self.stats['average_processing_time'] = processing_time
                        else:
                            self.stats['average_processing_time'] = (
                                self.stats['average_processing_time'] * 0.9 +
                                processing_time * 0.1
                            )
                        
                        logger.debug(f"Completed task {task.task_id} in {processing_time:.2f}s")
                        
                    except Exception as e:
                        task.status = "failed"
                        task.error = str(e)
                        self.stats['tasks_failed'] += 1
                        logger.error(f"Task {task.task_id} failed: {e}")
                else:
                    task.status = "failed"
                    task.error = f"No processor for task type: {task.task_type}"
                    self.stats['tasks_failed'] += 1
                    logger.error(f"No processor for task type: {task.task_type}")
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'node_id': self.node_id,
            'is_running': self._is_running,
            'worker_count': len(self._worker_threads),
            'queue_size': self.task_queue.get_queue_size(),
            'registered_processors': list(self.task_processors.keys())
        }


class DistributedAnalyzer:
    """Distributed analyzer for processing large MoE datasets."""
    
    def __init__(self, coordinator: DistributedCoordinator):
        self.coordinator = coordinator
        
        # Register processors
        self.coordinator.register_processor('analyze_routing_batch', self._process_routing_batch)
        self.coordinator.register_processor('compute_expert_stats', self._compute_expert_stats)
        self.coordinator.register_processor('load_balance_analysis', self._load_balance_analysis)
    
    def analyze_routing_events_distributed(self, events: List[RoutingEvent], 
                                         batch_size: int = 1000) -> Dict[str, Any]:
        """Analyze routing events using distributed processing."""
        if not events:
            return {}
        
        # Split events into batches
        batches = []
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            batches.append([event.__dict__ for event in batch])  # Serialize for processing
        
        # Submit batch processing tasks
        task_ids = []
        for i, batch in enumerate(batches):
            task_id = self.coordinator.submit_task(
                'analyze_routing_batch',
                {'batch': batch, 'batch_id': i},
                priority=1
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} batch analysis tasks")
        
        # For now, return a summary (in real implementation, you'd wait for results)
        return {
            'total_events': len(events),
            'batches_submitted': len(batches),
            'batch_size': batch_size,
            'task_ids': task_ids
        }
    
    def _process_routing_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of routing events."""
        batch_data = payload['batch']
        batch_id = payload['batch_id']
        
        # Convert back to RoutingEvent objects
        events = []
        for event_data in batch_data:
            event = RoutingEvent(**event_data)
            events.append(event)
        
        # Perform analysis
        expert_usage = {}
        layer_stats = {}
        confidence_scores = []
        
        for event in events:
            # Expert usage tracking
            for expert_id in event.selected_experts:
                expert_usage[expert_id] = expert_usage.get(expert_id, 0) + 1
            
            # Layer statistics
            layer_id = event.layer_idx
            if layer_id not in layer_stats:
                layer_stats[layer_id] = {'token_count': 0, 'expert_selections': []}
            layer_stats[layer_id]['token_count'] += 1
            layer_stats[layer_id]['expert_selections'].extend(event.selected_experts)
            
            # Confidence scores
            if event.confidence_scores:
                confidence_scores.extend(event.confidence_scores)
        
        return {
            'batch_id': batch_id,
            'event_count': len(events),
            'expert_usage': expert_usage,
            'layer_stats': layer_stats,
            'confidence_scores': confidence_scores,
            'processing_time': time.time()
        }
    
    def _compute_expert_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute expert utilization statistics."""
        # Placeholder for expert statistics computation
        return {'status': 'completed', 'expert_count': payload.get('expert_count', 0)}
    
    def _load_balance_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze load balancing across experts."""
        # Placeholder for load balance analysis
        return {'status': 'completed', 'balance_score': 0.85}


# Global distributed coordinator
_global_coordinator: Optional[DistributedCoordinator] = None


def get_global_coordinator(node_id: Optional[str] = None, 
                         redis_url: Optional[str] = None) -> DistributedCoordinator:
    """Get or create the global distributed coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = DistributedCoordinator(node_id, redis_url)
    return _global_coordinator


def start_distributed_processing(num_workers: int = None):
    """Start distributed processing on the global coordinator."""
    coordinator = get_global_coordinator()
    coordinator.start_processing(num_workers)


def stop_distributed_processing():
    """Stop distributed processing on the global coordinator."""
    global _global_coordinator
    if _global_coordinator:
        _global_coordinator.stop_processing()