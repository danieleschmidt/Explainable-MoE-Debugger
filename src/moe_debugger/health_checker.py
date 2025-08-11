"""Advanced health checking and system diagnostics."""

import asyncio
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    process_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.check_results: Dict[str, HealthCheckResult] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Default system health checks
        self._register_default_checks()
    
    def register_check(self, name: str, check_func: Callable[[], Tuple[HealthStatus, str, Dict]]):
        """Register a custom health check."""
        with self._lock:
            self.health_checks[name] = check_func
            logger.info(f"Registered health check: {name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                if name in self.check_results:
                    del self.check_results[name]
                logger.info(f"Unregistered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        
        try:
            status, message, metadata = self.health_checks[name]()
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata=metadata
            )
            
            with self._lock:
                self.check_results[name] = result
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms
            )
            
            with self._lock:
                self.check_results[name] = error_result
            
            logger.error(f"Health check '{name}' failed: {e}")
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name in list(self.health_checks.keys()):
            results[name] = self.run_check(name)
        
        return results
    
    async def run_all_checks_async(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks asynchronously."""
        tasks = []
        
        async def run_check_async(name: str) -> Tuple[str, HealthCheckResult]:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.run_check, name)
            return name, result
        
        for name in list(self.health_checks.keys()):
            tasks.append(run_check_async(name))
        
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"Async health check failed: {task_result}")
            else:
                name, result = task_result
                results[name] = result
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.check_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.check_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.WARNING
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run all health checks
                self.run_all_checks()
                
                # Collect system metrics
                metrics = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics.append(metrics)
                    
                    # Keep only recent metrics (last 24 hours)
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.system_metrics = [
                        m for m in self.system_metrics 
                        if m.timestamp > cutoff_time
                    ]
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                process_count=0
            )
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        def cpu_check() -> Tuple[HealthStatus, str, Dict]:
            """Check CPU usage."""
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if cpu_percent > 90:
                return HealthStatus.CRITICAL, f"CPU usage critical: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
            elif cpu_percent > 75:
                return HealthStatus.WARNING, f"CPU usage high: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
            else:
                return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
        
        def memory_check() -> Tuple[HealthStatus, str, Dict]:
            """Check memory usage."""
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return HealthStatus.CRITICAL, f"Memory usage critical: {memory.percent:.1f}%", {"memory_percent": memory.percent}
            elif memory.percent > 75:
                return HealthStatus.WARNING, f"Memory usage high: {memory.percent:.1f}%", {"memory_percent": memory.percent}
            else:
                return HealthStatus.HEALTHY, f"Memory usage normal: {memory.percent:.1f}%", {"memory_percent": memory.percent}
        
        def disk_check() -> Tuple[HealthStatus, str, Dict]:
            """Check disk usage."""
            try:
                disk = psutil.disk_usage('/')
                
                if disk.percent > 95:
                    return HealthStatus.CRITICAL, f"Disk usage critical: {disk.percent:.1f}%", {"disk_percent": disk.percent}
                elif disk.percent > 85:
                    return HealthStatus.WARNING, f"Disk usage high: {disk.percent:.1f}%", {"disk_percent": disk.percent}
                else:
                    return HealthStatus.HEALTHY, f"Disk usage normal: {disk.percent:.1f}%", {"disk_percent": disk.percent}
            except Exception:
                return HealthStatus.WARNING, "Could not check disk usage", {}
        
        def process_check() -> Tuple[HealthStatus, str, Dict]:
            """Check process count."""
            process_count = len(psutil.pids())
            
            if process_count > 1000:
                return HealthStatus.WARNING, f"High process count: {process_count}", {"process_count": process_count}
            else:
                return HealthStatus.HEALTHY, f"Process count normal: {process_count}", {"process_count": process_count}
        
        self.register_check("cpu", cpu_check)
        self.register_check("memory", memory_check)
        self.register_check("disk", disk_check)
        self.register_check("processes", process_check)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_status = self.get_overall_health()
        
        # Get recent system metrics
        recent_metrics = self.system_metrics[-10:] if self.system_metrics else []
        
        report = {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms,
                "metadata": result.metadata
            } for name, result in self.check_results.items()},
            "system_metrics": {
                "current": self._collect_system_metrics().__dict__ if recent_metrics else None,
                "history": [m.__dict__ for m in recent_metrics]
            },
            "monitoring": {
                "is_active": self.is_monitoring,
                "check_interval": self.check_interval,
                "total_checks": len(self.health_checks)
            }
        }
        
        return report
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(time.time() * 1000)
        
        # Overall health status (0=healthy, 1=warning, 2=critical, 3=unknown)
        status_map = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.CRITICAL: 2,
            HealthStatus.UNKNOWN: 3
        }
        
        overall_status = self.get_overall_health()
        lines.append(f'moe_debugger_health_overall {{}} {status_map[overall_status]} {timestamp}')
        
        # Individual check statuses
        for name, result in self.check_results.items():
            status_value = status_map[result.status]
            lines.append(f'moe_debugger_health_check{{name="{name}"}} {status_value} {timestamp}')
            lines.append(f'moe_debugger_health_check_duration_ms{{name="{name}"}} {result.duration_ms} {timestamp}')
        
        # System metrics
        if self.system_metrics:
            latest = self.system_metrics[-1]
            lines.append(f'moe_debugger_cpu_percent {{}} {latest.cpu_percent} {timestamp}')
            lines.append(f'moe_debugger_memory_percent {{}} {latest.memory_percent} {timestamp}')
            lines.append(f'moe_debugger_memory_available_mb {{}} {latest.memory_available_mb} {timestamp}')
            lines.append(f'moe_debugger_disk_usage_percent {{}} {latest.disk_usage_percent} {timestamp}')
            lines.append(f'moe_debugger_process_count {{}} {latest.process_count} {timestamp}')
        
        return '\n'.join(lines)


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker