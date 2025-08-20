"""
Prometheus metrics exporter for progressive quality gates monitoring.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response
import psutil
import threading
from datetime import datetime


class ProgressiveQualityGatesMetrics:
    """Prometheus metrics exporter for progressive quality gates."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._start_background_collection()
    
    def _setup_metrics(self):
        """Setup all Prometheus metrics."""
        
        # API Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # MoE Debugger Specific Metrics
        self.routing_events_total = Counter(
            'moe_debugger_routing_events_total',
            'Total number of routing events processed',
            ['expert_id', 'layer_id'],
            registry=self.registry
        )
        
        self.routing_events_backlog = Gauge(
            'moe_debugger_routing_events_backlog',
            'Number of routing events waiting to be processed',
            registry=self.registry
        )
        
        self.expert_utilization = Gauge(
            'moe_debugger_expert_utilization',
            'Expert utilization percentage',
            ['expert_id'],
            registry=self.registry
        )
        
        self.dead_experts_count = Gauge(
            'moe_debugger_dead_experts_count',
            'Number of dead experts (no routing activity)',
            registry=self.registry
        )
        
        self.sessions_active = Gauge(
            'moe_debugger_sessions_active',
            'Number of active debugging sessions',
            registry=self.registry
        )
        
        self.sessions_created_total = Counter(
            'moe_debugger_sessions_created_total',
            'Total number of sessions created',
            registry=self.registry
        )
        
        self.sessions_failed_total = Counter(
            'moe_debugger_sessions_failed_total',
            'Total number of failed session creations',
            ['reason'],
            registry=self.registry
        )
        
        # WebSocket Metrics
        self.websocket_connections_active = Gauge(
            'websocket_connections_active',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_connections_total = Counter(
            'websocket_connections_total',
            'Total number of WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_connections_failed_total = Counter(
            'websocket_connections_failed_total',
            'Total number of failed WebSocket connections',
            ['reason'],
            registry=self.registry
        )
        
        self.websocket_messages_total = Counter(
            'websocket_messages_total',
            'Total number of WebSocket messages',
            ['direction', 'type'],
            registry=self.registry
        )
        
        self.websocket_message_duration_seconds = Histogram(
            'websocket_message_duration_seconds',
            'WebSocket message processing duration',
            ['type'],
            registry=self.registry
        )
        
        # Quality Gates Metrics
        self.quality_gate_status = Gauge(
            'quality_gate_status',
            'Quality gate status (1=pass, 0=fail)',
            ['gate_type'],
            registry=self.registry
        )
        
        self.test_coverage_percentage = Gauge(
            'test_coverage_percentage',
            'Test coverage percentage',
            registry=self.registry
        )
        
        self.code_quality_score = Gauge(
            'code_quality_score',
            'Code quality score (0-100)',
            registry=self.registry
        )
        
        self.security_scan_score = Gauge(
            'security_scan_score',
            'Security scan score (0-100)',
            registry=self.registry
        )
        
        self.security_vulnerabilities = Gauge(
            'security_vulnerabilities',
            'Number of security vulnerabilities',
            ['severity'],
            registry=self.registry
        )
        
        self.performance_score = Gauge(
            'performance_score',
            'Performance score (0-100)',
            registry=self.registry
        )
        
        self.performance_regression_detected = Gauge(
            'performance_regression_detected',
            'Performance regression detected (1=yes, 0=no)',
            registry=self.registry
        )
        
        # Health Check Metrics
        self.health_check_status = Gauge(
            'health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['check_name'],
            registry=self.registry
        )
        
        self.health_check_duration_seconds = Histogram(
            'health_check_duration_seconds',
            'Health check duration in seconds',
            ['check_name'],
            registry=self.registry
        )
        
        self.health_check_failures_consecutive = Gauge(
            'health_check_failures_consecutive',
            'Number of consecutive health check failures',
            registry=self.registry
        )
        
        # Deployment Metrics
        self.deployment_info = Info(
            'deployment_info',
            'Deployment information',
            registry=self.registry
        )
        
        self.deployment_timestamp = Gauge(
            'deployment_timestamp',
            'Timestamp of last deployment',
            registry=self.registry
        )
        
        self.auto_rollback_triggered_total = Counter(
            'auto_rollback_triggered_total',
            'Total number of automatic rollbacks triggered',
            ['reason'],
            registry=self.registry
        )
        
        # System Metrics
        self.system_cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['mountpoint'],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_operations_duration_seconds = Histogram(
            'cache_operations_duration_seconds',
            'Cache operation duration in seconds',
            ['operation', 'cache_type'],
            registry=self.registry
        )
    
    def _start_background_collection(self):
        """Start background thread for collecting system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage_percent.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_memory_usage_percent.set(memory.percent)
                    
                    # Disk usage
                    for partition in psutil.disk_partitions():
                        try:
                            disk_usage = psutil.disk_usage(partition.mountpoint)
                            usage_percent = (disk_usage.used / disk_usage.total) * 100
                            self.system_disk_usage_percent.labels(
                                mountpoint=partition.mountpoint
                            ).set(usage_percent)
                        except (PermissionError, FileNotFoundError):
                            continue
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    # API Metrics Methods
    def record_http_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    # MoE Debugger Metrics Methods
    def record_routing_event(self, expert_id: int, layer_id: int):
        """Record routing event processing."""
        self.routing_events_total.labels(expert_id=str(expert_id), layer_id=str(layer_id)).inc()
    
    def update_routing_backlog(self, count: int):
        """Update routing events backlog count."""
        self.routing_events_backlog.set(count)
    
    def update_expert_utilization(self, expert_id: int, utilization: float):
        """Update expert utilization percentage."""
        self.expert_utilization.labels(expert_id=str(expert_id)).set(utilization * 100)
    
    def update_dead_experts_count(self, count: int):
        """Update count of dead experts."""
        self.dead_experts_count.set(count)
    
    def update_active_sessions(self, count: int):
        """Update count of active sessions."""
        self.sessions_active.set(count)
    
    def record_session_created(self):
        """Record successful session creation."""
        self.sessions_created_total.inc()
    
    def record_session_failed(self, reason: str):
        """Record failed session creation."""
        self.sessions_failed_total.labels(reason=reason).inc()
    
    # WebSocket Metrics Methods
    def update_websocket_connections(self, active: int):
        """Update active WebSocket connections count."""
        self.websocket_connections_active.set(active)
    
    def record_websocket_connection(self):
        """Record new WebSocket connection."""
        self.websocket_connections_total.inc()
    
    def record_websocket_connection_failed(self, reason: str):
        """Record failed WebSocket connection."""
        self.websocket_connections_failed_total.labels(reason=reason).inc()
    
    def record_websocket_message(self, direction: str, message_type: str, duration: float):
        """Record WebSocket message."""
        self.websocket_messages_total.labels(direction=direction, type=message_type).inc()
        self.websocket_message_duration_seconds.labels(type=message_type).observe(duration)
    
    # Quality Gates Metrics Methods
    def update_quality_gate_status(self, gate_type: str, passed: bool):
        """Update quality gate status."""
        self.quality_gate_status.labels(gate_type=gate_type).set(1 if passed else 0)
    
    def update_test_coverage(self, percentage: float):
        """Update test coverage percentage."""
        self.test_coverage_percentage.set(percentage)
    
    def update_code_quality_score(self, score: float):
        """Update code quality score."""
        self.code_quality_score.set(score)
    
    def update_security_scan_score(self, score: float):
        """Update security scan score."""
        self.security_scan_score.set(score)
    
    def update_security_vulnerabilities(self, severity: str, count: int):
        """Update security vulnerabilities count."""
        self.security_vulnerabilities.labels(severity=severity).set(count)
    
    def update_performance_score(self, score: float):
        """Update performance score."""
        self.performance_score.set(score)
    
    def set_performance_regression(self, detected: bool):
        """Set performance regression detection flag."""
        self.performance_regression_detected.set(1 if detected else 0)
    
    # Health Check Metrics Methods
    def record_health_check(self, check_name: str, healthy: bool, duration: float):
        """Record health check result."""
        self.health_check_status.labels(check_name=check_name).set(1 if healthy else 0)
        self.health_check_duration_seconds.labels(check_name=check_name).observe(duration)
    
    def update_consecutive_failures(self, count: int):
        """Update consecutive health check failures count."""
        self.health_check_failures_consecutive.set(count)
    
    # Deployment Metrics Methods
    def update_deployment_info(self, version: str, timestamp: str, environment: str):
        """Update deployment information."""
        self.deployment_info.info({
            'version': version,
            'timestamp': timestamp,
            'environment': environment
        })
        self.deployment_timestamp.set(time.time())
    
    def record_auto_rollback(self, reason: str):
        """Record automatic rollback trigger."""
        self.auto_rollback_triggered_total.labels(reason=reason).inc()
    
    # Cache Metrics Methods
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def record_cache_operation(self, operation: str, cache_type: str, duration: float):
        """Record cache operation duration."""
        self.cache_operations_duration_seconds.labels(
            operation=operation, 
            cache_type=cache_type
        ).observe(duration)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_response(self) -> Response:
        """Get metrics as FastAPI response."""
        return Response(
            content=generate_latest(self.registry),
            media_type=CONTENT_TYPE_LATEST
        )


# Global metrics instance
metrics = ProgressiveQualityGatesMetrics()


def get_metrics_instance() -> ProgressiveQualityGatesMetrics:
    """Get the global metrics instance."""
    return metrics


# Decorator for timing functions
def timed_metric(metric_histogram, labels=None):
    """Decorator to time function execution and record in histogram."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric_histogram.labels(**labels).observe(duration)
                else:
                    metric_histogram.observe(duration)
        return wrapper
    return decorator


# Context manager for timing code blocks
class MetricTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, metric_histogram, labels=None):
        self.metric_histogram = metric_histogram
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            if self.labels:
                self.metric_histogram.labels(**self.labels).observe(duration)
            else:
                self.metric_histogram.observe(duration)