"""System monitoring and health checks for MoE debugger."""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging_config import get_logger
from .validation import safe_json_dumps

logger = get_logger(__name__)


class HealthCheck:
    """Individual health check definition."""
    
    def __init__(self, name: str, check_func: Callable[[], bool], 
                 critical: bool = False, timeout: float = 5.0):
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.timeout = timeout
        self.last_result = None
        self.last_check_time = None
        self.failure_count = 0
        self.success_count = 0
    
    def run_check(self) -> Dict[str, Any]:
        """Run the health check and return results."""
        start_time = time.perf_counter()
        
        try:
            # Run check with timeout
            result = self._run_with_timeout()
            duration = time.perf_counter() - start_time
            
            self.last_result = result
            self.last_check_time = time.time()
            
            if result:
                self.success_count += 1
                status = "healthy"
                logger.debug(f"Health check '{self.name}' passed in {duration:.3f}s")
            else:
                self.failure_count += 1
                status = "unhealthy"
                logger.warning(f"Health check '{self.name}' failed in {duration:.3f}s")
            
            return {
                "name": self.name,
                "status": status,
                "critical": self.critical,
                "duration": duration,
                "timestamp": self.last_check_time,
                "failure_count": self.failure_count,
                "success_count": self.success_count
            }
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.failure_count += 1
            self.last_result = False
            self.last_check_time = time.time()
            
            logger.error(f"Health check '{self.name}' error: {e}")
            
            return {
                "name": self.name,
                "status": "error",
                "critical": self.critical,
                "duration": duration,
                "timestamp": self.last_check_time,
                "error": str(e),
                "failure_count": self.failure_count,
                "success_count": self.success_count
            }
    
    def _run_with_timeout(self) -> bool:
        """Run check function with timeout."""
        if self.timeout <= 0:
            return self.check_func()
        
        result = [False]  # Mutable container for thread communication
        exception = [None]
        
        def target():
            try:
                result[0] = self.check_func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Health check timed out after {self.timeout}s")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        self._last_network_stats = None
        self.running = False
        self.monitor_thread = None
        self._lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring in background thread."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """Collect system metrics."""
        timestamp = time.time()
        
        with self._lock:
            # CPU metrics
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                self.cpu_history.append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'load_avg': getattr(psutil, 'getloadavg', lambda: [0, 0, 0])()
                })
                
                self.memory_history.append({
                    'timestamp': timestamp,
                    'total_mb': memory.total / 1024 / 1024,
                    'available_mb': memory.available / 1024 / 1024,
                    'used_mb': memory.used / 1024 / 1024,
                    'percent': memory.percent
                })
                
                self.disk_history.append({
                    'timestamp': timestamp,
                    'total_gb': disk.total / 1024 / 1024 / 1024,
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'free_gb': disk.free / 1024 / 1024 / 1024,
                    'percent': (disk.used / disk.total) * 100
                })
                
                # Network delta metrics
                if self._last_network_stats:
                    time_delta = timestamp - self._last_network_stats['timestamp']
                    bytes_sent_delta = network.bytes_sent - self._last_network_stats['bytes_sent']
                    bytes_recv_delta = network.bytes_recv - self._last_network_stats['bytes_recv']
                    
                    self.network_history.append({
                        'timestamp': timestamp,
                        'bytes_sent_per_sec': bytes_sent_delta / time_delta if time_delta > 0 else 0,
                        'bytes_recv_per_sec': bytes_recv_delta / time_delta if time_delta > 0 else 0,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    })
                
                self._last_network_stats = {
                    'timestamp': timestamp,
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
            else:
                # Fallback metrics when psutil not available
                self.cpu_history.append({
                    'timestamp': timestamp,
                    'cpu_percent': 0.0,
                    'load_avg': [0, 0, 0]
                })
                
                self.memory_history.append({
                    'timestamp': timestamp,
                    'total_mb': 0,
                    'available_mb': 0,
                    'used_mb': 0,
                    'percent': 0.0
                })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            current_metrics = {
                'timestamp': time.time(),
                'monitoring_active': self.running
            }
            
            # Latest metrics
            if self.cpu_history:
                current_metrics['cpu'] = self.cpu_history[-1]
            if self.memory_history:
                current_metrics['memory'] = self.memory_history[-1]
            if self.disk_history:
                current_metrics['disk'] = self.disk_history[-1]
            if self.network_history:
                current_metrics['network'] = self.network_history[-1]
            
            return current_metrics
    
    def get_historical_metrics(self, minutes: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics for specified time period."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            def filter_recent(history):
                return [item for item in history if item['timestamp'] >= cutoff_time]
            
            return {
                'cpu': filter_recent(self.cpu_history),
                'memory': filter_recent(self.memory_history),
                'disk': filter_recent(self.disk_history),
                'network': filter_recent(self.network_history)
            }
    
    def get_statistics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get statistical summary of metrics."""
        historical = self.get_historical_metrics(minutes)
        stats = {}
        
        # CPU statistics
        if historical['cpu']:
            cpu_values = [item['cpu_percent'] for item in historical['cpu']]
            stats['cpu'] = {
                'mean': statistics.mean(cpu_values),
                'median': statistics.median(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            }
        
        # Memory statistics
        if historical['memory']:
            memory_values = [item['percent'] for item in historical['memory']]
            stats['memory'] = {
                'mean_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'current_used_mb': historical['memory'][-1]['used_mb'] if historical['memory'] else 0
            }
        
        # Disk statistics
        if historical['disk']:
            disk_values = [item['percent'] for item in historical['disk']]
            stats['disk'] = {
                'mean_percent': statistics.mean(disk_values),
                'current_percent': disk_values[-1] if disk_values else 0,
                'current_free_gb': historical['disk'][-1]['free_gb'] if historical['disk'] else 0
            }
        
        return stats


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_monitor = SystemMonitor()
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'disk_percent': 90.0,
            'critical_health_checks_failed': 1
        }
        self.alerts_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_health_check(self, name: str, check_func: Callable[[], bool], 
                        critical: bool = False, timeout: float = 5.0):
        """Add a health check."""
        with self._lock:
            self.health_checks[name] = HealthCheck(name, check_func, critical, timeout)
        logger.info(f"Added health check: {name} (critical: {critical})")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Removed health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': [],
            'summary': {
                'total': 0,
                'healthy': 0,
                'unhealthy': 0,
                'errors': 0,
                'critical_failed': 0
            }
        }
        
        with self._lock:
            for health_check in self.health_checks.values():
                check_result = health_check.run_check()
                results['checks'].append(check_result)
                
                # Update summary
                results['summary']['total'] += 1
                
                if check_result['status'] == 'healthy':
                    results['summary']['healthy'] += 1
                elif check_result['status'] == 'unhealthy':
                    results['summary']['unhealthy'] += 1
                    if check_result['critical']:
                        results['summary']['critical_failed'] += 1
                else:  # error
                    results['summary']['errors'] += 1
                    if check_result['critical']:
                        results['summary']['critical_failed'] += 1
        
        # Determine overall status
        if results['summary']['critical_failed'] > 0:
            results['overall_status'] = 'critical'
        elif results['summary']['unhealthy'] > 0 or results['summary']['errors'] > 0:
            results['overall_status'] = 'degraded'
        
        # Check for alerts
        self._check_alerts(results)
        
        return results
    
    def _check_alerts(self, health_results: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        # Check critical health checks
        if health_results['summary']['critical_failed'] >= self.alert_thresholds['critical_health_checks_failed']:
            alerts.append({
                'type': 'critical_health_check_failed',
                'message': f"{health_results['summary']['critical_failed']} critical health checks failed",
                'severity': 'critical'
            })
        
        # Check system metrics
        current_metrics = self.system_monitor.get_current_metrics()
        
        if 'cpu' in current_metrics:
            cpu_percent = current_metrics['cpu']['cpu_percent']
            if cpu_percent >= self.alert_thresholds['cpu_percent']:
                alerts.append({
                    'type': 'high_cpu_usage',
                    'message': f"CPU usage at {cpu_percent:.1f}%",
                    'severity': 'warning'
                })
        
        if 'memory' in current_metrics:
            memory_percent = current_metrics['memory']['percent']
            if memory_percent >= self.alert_thresholds['memory_percent']:
                alerts.append({
                    'type': 'high_memory_usage',
                    'message': f"Memory usage at {memory_percent:.1f}%",
                    'severity': 'warning'
                })
        
        if 'disk' in current_metrics:
            disk_percent = current_metrics['disk']['percent']
            if disk_percent >= self.alert_thresholds['disk_percent']:
                alerts.append({
                    'type': 'high_disk_usage',
                    'message': f"Disk usage at {disk_percent:.1f}%",
                    'severity': 'warning'
                })
        
        # Store alerts
        timestamp = time.time()
        for alert in alerts:
            alert['timestamp'] = timestamp
            self.alerts_history.append(alert)
            
            # Log alerts
            if alert['severity'] == 'critical':
                logger.critical(f"ALERT: {alert['message']}")
            else:
                logger.warning(f"ALERT: {alert['message']}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = self.run_health_checks()
        system_metrics = self.system_monitor.get_current_metrics()
        system_stats = self.system_monitor.get_statistics()
        
        return {
            'timestamp': time.time(),
            'health': health_results,
            'system_metrics': system_metrics,
            'system_statistics': system_stats,
            'recent_alerts': list(self.alerts_history)[-10:],  # Last 10 alerts
            'alert_thresholds': self.alert_thresholds
        }
    
    def start_monitoring(self, health_check_interval: float = 30.0, 
                        system_monitor_interval: float = 5.0):
        """Start all monitoring."""
        self.system_monitor.start_monitoring(system_monitor_interval)
        
        # Start health check scheduler
        def health_check_loop():
            while True:
                try:
                    self.run_health_checks()
                    time.sleep(health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    time.sleep(health_check_interval)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring."""
        self.system_monitor.stop_monitoring()
        logger.info("Health monitoring stopped")


# Default health checks
def create_default_health_checks(health_monitor: HealthMonitor, debugger=None):
    """Create default health checks for the debugger."""
    
    def check_memory_available():
        """Check if sufficient memory is available."""
        if not PSUTIL_AVAILABLE:
            return True
        memory = psutil.virtual_memory()
        return memory.available > 1024 * 1024 * 1024  # 1GB available
    
    def check_disk_space():
        """Check if sufficient disk space is available."""
        if not PSUTIL_AVAILABLE:
            return True
        disk = psutil.disk_usage('/')
        return disk.free > 5 * 1024 * 1024 * 1024  # 5GB free
    
    def check_debugger_status():
        """Check if debugger is responsive."""
        if not debugger:
            return True
        try:
            # Simple check - see if we can get model summary
            debugger.get_routing_stats()
            return True
        except Exception:
            return False
    
    # Add health checks
    health_monitor.add_health_check("memory_available", check_memory_available, critical=True)
    health_monitor.add_health_check("disk_space", check_disk_space, critical=True)
    
    if debugger:
        health_monitor.add_health_check("debugger_responsive", check_debugger_status, critical=False)


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    global _health_monitor
    
    with _monitor_lock:
        if _health_monitor is None:
            _health_monitor = HealthMonitor()
        return _health_monitor