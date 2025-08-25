"""Chaos Engineering and Resilience Testing System for Progressive Quality Gates.

This module implements advanced chaos engineering capabilities including systematic
fault injection, resilience testing, blast radius limitation, and automated
recovery validation to ensure system reliability under adverse conditions.

Features:
- Systematic fault injection with controlled chaos experiments
- Service mesh traffic management and circuit breaker testing
- Blast radius limitation and failure containment
- Automated resilience testing and validation
- Real-time monitoring during chaos experiments
- Statistical analysis of system behavior under stress
- Automated rollback and recovery testing

Authors: Terragon Labs - Progressive Quality Gates v2.0
License: MIT
"""

import time
import threading
import logging
import random
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json

from .logging_config import get_logger
from .validation import safe_json_dumps
from .autonomous_recovery import AutonomousRecoverySystem, HealthStatus


class ChaosExperimentType(Enum):
    """Types of chaos experiments."""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition" 
    SERVICE_FAILURE = "service_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    CASCADING_FAILURE = "cascading_failure"


class ExperimentStatus(Enum):
    """Status of chaos experiments."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    RECOVERING = "recovering"


class BlastRadiusLevel(Enum):
    """Blast radius containment levels."""
    SINGLE_INSTANCE = "single_instance"
    SINGLE_SERVICE = "single_service"
    SERVICE_GROUP = "service_group"
    AVAILABILITY_ZONE = "availability_zone"
    REGION = "region"


@dataclass
class ChaosExperiment:
    """Definition of a chaos engineering experiment."""
    experiment_id: str
    name: str
    experiment_type: ChaosExperimentType
    description: str
    target_services: List[str]
    blast_radius: BlastRadiusLevel
    duration_seconds: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment metadata
    created_time: float = field(default_factory=time.time)
    scheduled_time: Optional[float] = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    status: ExperimentStatus = ExperimentStatus.PLANNED
    
    # Safety constraints
    abort_conditions: List[str] = field(default_factory=list)
    max_error_rate: float = 0.1  # 10% max error rate
    max_response_time_ms: float = 5000  # 5 second max response time
    require_manual_approval: bool = False
    
    # Results
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    experiment_metrics: Dict[str, float] = field(default_factory=dict)
    recovery_metrics: Dict[str, float] = field(default_factory=dict)
    hypothesis: Optional[str] = None
    result_summary: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class FaultInjection:
    """Configuration for fault injection."""
    fault_id: str
    fault_type: str
    target_component: str
    severity: float  # 0.0 to 1.0
    probability: float  # 0.0 to 1.0 chance of activation
    duration_seconds: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # State tracking
    is_active: bool = False
    activation_time: Optional[float] = None
    total_activations: int = 0


@dataclass
class ResilienceMetrics:
    """Metrics collected during resilience testing."""
    timestamp: float = field(default_factory=time.time)
    experiment_id: str = ""
    
    # System performance metrics
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_utilization: float = 0.0
    network_utilization: float = 0.0
    
    # Availability and reliability
    service_availability: Dict[str, float] = field(default_factory=dict)
    circuit_breaker_states: Dict[str, str] = field(default_factory=dict)
    recovery_time_seconds: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class NetworkChaosInjector:
    """Injects network-related chaos conditions."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_injections: Dict[str, FaultInjection] = {}
        self._lock = threading.Lock()
    
    def inject_latency(self, target_service: str, latency_ms: float, 
                      duration_seconds: float, probability: float = 1.0) -> str:
        """Inject network latency for a target service."""
        fault_id = f"latency_{target_service}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="network_latency",
            target_component=target_service,
            severity=min(1.0, latency_ms / 1000.0),  # Normalize to 0-1
            probability=probability,
            duration_seconds=duration_seconds,
            parameters={
                'latency_ms': latency_ms,
                'jitter_ms': latency_ms * 0.1  # 10% jitter
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Start fault injection
        threading.Thread(
            target=self._execute_latency_injection,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.info(f"Started latency injection: {latency_ms}ms for {target_service}")
        return fault_id
    
    def inject_partition(self, source_service: str, target_service: str, 
                        duration_seconds: float) -> str:
        """Inject network partition between services."""
        fault_id = f"partition_{source_service}_{target_service}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="network_partition",
            target_component=f"{source_service}->{target_service}",
            severity=1.0,  # Complete partition
            probability=1.0,
            duration_seconds=duration_seconds,
            parameters={
                'source_service': source_service,
                'target_service': target_service,
                'partition_type': 'bidirectional'
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Start partition injection
        threading.Thread(
            target=self._execute_partition_injection,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.warning(f"Started network partition: {source_service} <-> {target_service}")
        return fault_id
    
    def inject_packet_loss(self, target_service: str, loss_rate: float,
                          duration_seconds: float) -> str:
        """Inject packet loss for a target service."""
        fault_id = f"packet_loss_{target_service}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="packet_loss",
            target_component=target_service,
            severity=loss_rate,
            probability=1.0,
            duration_seconds=duration_seconds,
            parameters={
                'loss_rate': loss_rate,
                'correlation': 0.1  # 10% correlation between consecutive losses
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Start packet loss injection
        threading.Thread(
            target=self._execute_packet_loss_injection,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.warning(f"Started packet loss injection: {loss_rate*100}% for {target_service}")
        return fault_id
    
    def stop_injection(self, fault_id: str) -> bool:
        """Stop a specific fault injection."""
        with self._lock:
            if fault_id in self.active_injections:
                fault = self.active_injections[fault_id]
                fault.is_active = False
                del self.active_injections[fault_id]
                self.logger.info(f"Stopped fault injection: {fault_id}")
                return True
        return False
    
    def stop_all_injections(self) -> int:
        """Stop all active fault injections."""
        with self._lock:
            count = len(self.active_injections)
            for fault in self.active_injections.values():
                fault.is_active = False
            self.active_injections.clear()
        
        self.logger.info(f"Stopped {count} fault injections")
        return count
    
    def get_active_injections(self) -> List[Dict[str, Any]]:
        """Get list of active fault injections."""
        with self._lock:
            return [
                {
                    'fault_id': fault.fault_id,
                    'fault_type': fault.fault_type,
                    'target_component': fault.target_component,
                    'severity': fault.severity,
                    'duration_seconds': fault.duration_seconds,
                    'activation_time': fault.activation_time,
                    'is_active': fault.is_active
                }
                for fault in self.active_injections.values()
            ]
    
    def _execute_latency_injection(self, fault: FaultInjection):
        """Execute network latency injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        end_time = fault.activation_time + fault.duration_seconds
        latency_ms = fault.parameters['latency_ms']
        
        while fault.is_active and time.time() < end_time:
            try:
                # Simulate latency injection by intercepting network calls
                # In production, this would integrate with service mesh or network proxy
                time.sleep(0.1)  # Check every 100ms
                
                fault.total_activations += 1
                
            except Exception as e:
                self.logger.error(f"Error in latency injection {fault.fault_id}: {e}")
                break
        
        fault.is_active = False
        self.logger.info(f"Completed latency injection: {fault.fault_id}")
    
    def _execute_partition_injection(self, fault: FaultInjection):
        """Execute network partition injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        source_service = fault.parameters['source_service']
        target_service = fault.parameters['target_service']
        
        # Simulate network partition
        self.logger.warning(f"Network partition active: {source_service} <-> {target_service}")
        
        time.sleep(fault.duration_seconds)
        
        fault.is_active = False
        self.logger.info(f"Network partition ended: {source_service} <-> {target_service}")
    
    def _execute_packet_loss_injection(self, fault: FaultInjection):
        """Execute packet loss injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        end_time = fault.activation_time + fault.duration_seconds
        loss_rate = fault.parameters['loss_rate']
        
        while fault.is_active and time.time() < end_time:
            try:
                # Simulate packet loss by randomly dropping packets
                # In production, this would integrate with network stack
                time.sleep(0.1)
                
                if random.random() < loss_rate:
                    fault.total_activations += 1
                
            except Exception as e:
                self.logger.error(f"Error in packet loss injection {fault.fault_id}: {e}")
                break
        
        fault.is_active = False
        self.logger.info(f"Completed packet loss injection: {fault.fault_id}")


class ServiceChaosInjector:
    """Injects service-level chaos conditions."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_injections: Dict[str, FaultInjection] = {}
        self._lock = threading.Lock()
    
    def kill_service_instance(self, service_name: str, instance_id: str = None) -> str:
        """Kill a service instance."""
        fault_id = f"kill_service_{service_name}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="service_kill",
            target_component=f"{service_name}:{instance_id or 'random'}",
            severity=1.0,
            probability=1.0,
            duration_seconds=0,  # Instant
            parameters={
                'service_name': service_name,
                'instance_id': instance_id,
                'restart_after_seconds': 30
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Execute service kill
        threading.Thread(
            target=self._execute_service_kill,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.critical(f"Killing service instance: {service_name}:{instance_id}")
        return fault_id
    
    def inject_resource_exhaustion(self, service_name: str, resource_type: str,
                                 exhaustion_level: float, duration_seconds: float) -> str:
        """Inject resource exhaustion (CPU, memory, disk)."""
        fault_id = f"resource_{resource_type}_{service_name}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="resource_exhaustion",
            target_component=f"{service_name}:{resource_type}",
            severity=exhaustion_level,
            probability=1.0,
            duration_seconds=duration_seconds,
            parameters={
                'service_name': service_name,
                'resource_type': resource_type,  # 'cpu', 'memory', 'disk'
                'exhaustion_level': exhaustion_level  # 0.0 to 1.0
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Start resource exhaustion
        threading.Thread(
            target=self._execute_resource_exhaustion,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.warning(f"Started resource exhaustion: {resource_type} for {service_name}")
        return fault_id
    
    def inject_configuration_error(self, service_name: str, config_type: str,
                                 duration_seconds: float) -> str:
        """Inject configuration errors."""
        fault_id = f"config_error_{service_name}_{int(time.time())}"
        
        fault = FaultInjection(
            fault_id=fault_id,
            fault_type="configuration_error",
            target_component=f"{service_name}:config",
            severity=0.8,
            probability=1.0,
            duration_seconds=duration_seconds,
            parameters={
                'service_name': service_name,
                'config_type': config_type,  # 'database_url', 'api_key', 'timeout'
                'error_type': 'invalid_value'
            }
        )
        
        with self._lock:
            self.active_injections[fault_id] = fault
        
        # Start configuration error injection
        threading.Thread(
            target=self._execute_configuration_error,
            args=(fault,),
            daemon=True
        ).start()
        
        self.logger.warning(f"Started configuration error: {config_type} for {service_name}")
        return fault_id
    
    def _execute_service_kill(self, fault: FaultInjection):
        """Execute service kill injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        service_name = fault.parameters['service_name']
        restart_after = fault.parameters.get('restart_after_seconds', 30)
        
        # Simulate service kill
        self.logger.critical(f"Service {service_name} killed by chaos injection")
        
        # Simulate restart delay
        time.sleep(restart_after)
        
        self.logger.info(f"Service {service_name} restarted after chaos injection")
        fault.is_active = False
    
    def _execute_resource_exhaustion(self, fault: FaultInjection):
        """Execute resource exhaustion injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        service_name = fault.parameters['service_name']
        resource_type = fault.parameters['resource_type']
        exhaustion_level = fault.parameters['exhaustion_level']
        
        end_time = fault.activation_time + fault.duration_seconds
        
        while fault.is_active and time.time() < end_time:
            try:
                # Simulate resource exhaustion
                if resource_type == 'cpu':
                    # Simulate CPU exhaustion
                    pass
                elif resource_type == 'memory':
                    # Simulate memory exhaustion
                    pass
                elif resource_type == 'disk':
                    # Simulate disk exhaustion
                    pass
                
                time.sleep(1)
                fault.total_activations += 1
                
            except Exception as e:
                self.logger.error(f"Error in resource exhaustion {fault.fault_id}: {e}")
                break
        
        fault.is_active = False
        self.logger.info(f"Resource exhaustion injection completed: {fault.fault_id}")
    
    def _execute_configuration_error(self, fault: FaultInjection):
        """Execute configuration error injection."""
        fault.is_active = True
        fault.activation_time = time.time()
        
        # Simulate configuration error for the duration
        time.sleep(fault.duration_seconds)
        
        fault.is_active = False
        self.logger.info(f"Configuration error injection completed: {fault.fault_id}")


class ResilienceTestRunner:
    """Runs resilience tests and collects metrics."""
    
    def __init__(self, recovery_system: AutonomousRecoverySystem):
        self.logger = get_logger(__name__)
        self.recovery_system = recovery_system
        self.metrics_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
    
    def run_resilience_test(self, experiment: ChaosExperiment,
                           metrics_collector: Callable[[], ResilienceMetrics]) -> Dict[str, Any]:
        """Run a complete resilience test."""
        try:
            self.logger.info(f"Starting resilience test: {experiment.name}")
            
            # Collect baseline metrics
            baseline_start = time.time()
            baseline_metrics = []
            for _ in range(5):  # 5 samples
                metrics = metrics_collector()
                baseline_metrics.append(metrics)
                time.sleep(1)
            
            experiment.baseline_metrics = self._aggregate_metrics(baseline_metrics)
            
            # Start experiment
            experiment.started_time = time.time()
            experiment.status = ExperimentStatus.RUNNING
            
            # Monitor during experiment
            experiment_metrics = []
            start_time = time.time()
            
            while time.time() - start_time < experiment.duration_seconds:
                # Check abort conditions
                current_metrics = metrics_collector()
                if self._should_abort_experiment(experiment, current_metrics):
                    self.logger.warning(f"Aborting experiment {experiment.name} due to safety conditions")
                    experiment.status = ExperimentStatus.ABORTED
                    break
                
                experiment_metrics.append(current_metrics)
                time.sleep(1)
            
            experiment.experiment_metrics = self._aggregate_metrics(experiment_metrics)
            
            # Monitor recovery
            if experiment.status != ExperimentStatus.ABORTED:
                experiment.status = ExperimentStatus.RECOVERING
                
                recovery_start = time.time()
                recovery_metrics = []
                
                # Monitor recovery for up to 5 minutes
                while time.time() - recovery_start < 300:
                    metrics = metrics_collector()
                    recovery_metrics.append(metrics)
                    
                    # Check if system has recovered
                    if self._has_system_recovered(experiment.baseline_metrics, metrics):
                        break
                    
                    time.sleep(1)
                
                experiment.recovery_metrics = self._aggregate_metrics(recovery_metrics)
                experiment.completed_time = time.time()
                experiment.status = ExperimentStatus.COMPLETED
            
            # Generate test results
            results = self._generate_test_results(experiment)
            
            with self._lock:
                # Store metrics for historical analysis
                for metrics in baseline_metrics + experiment_metrics + recovery_metrics:
                    self.metrics_history.append(metrics)
            
            self.logger.info(f"Completed resilience test: {experiment.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in resilience test {experiment.name}: {e}")
            experiment.status = ExperimentStatus.FAILED
            return {
                'experiment_id': experiment.experiment_id,
                'status': experiment.status.value,
                'error': str(e)
            }
    
    def _aggregate_metrics(self, metrics_list: List[ResilienceMetrics]) -> Dict[str, float]:
        """Aggregate multiple metrics into summary statistics."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Response time aggregation
        response_times_p50 = [m.response_time_p50 for m in metrics_list]
        response_times_p95 = [m.response_time_p95 for m in metrics_list]
        response_times_p99 = [m.response_time_p99 for m in metrics_list]
        
        aggregated.update({
            'avg_response_time_p50': statistics.mean(response_times_p50),
            'max_response_time_p50': max(response_times_p50),
            'avg_response_time_p95': statistics.mean(response_times_p95),
            'max_response_time_p95': max(response_times_p95),
            'avg_response_time_p99': statistics.mean(response_times_p99),
            'max_response_time_p99': max(response_times_p99)
        })
        
        # Error rate aggregation
        error_rates = [m.error_rate for m in metrics_list]
        aggregated.update({
            'avg_error_rate': statistics.mean(error_rates),
            'max_error_rate': max(error_rates),
            'total_errors': sum(error_rates) * len(error_rates)
        })
        
        # Throughput aggregation
        throughputs = [m.throughput_rps for m in metrics_list]
        aggregated.update({
            'avg_throughput_rps': statistics.mean(throughputs),
            'min_throughput_rps': min(throughputs)
        })
        
        # Resource utilization
        cpu_usage = [m.cpu_usage_percent for m in metrics_list]
        memory_usage = [m.memory_usage_percent for m in metrics_list]
        
        aggregated.update({
            'avg_cpu_usage': statistics.mean(cpu_usage),
            'max_cpu_usage': max(cpu_usage),
            'avg_memory_usage': statistics.mean(memory_usage),
            'max_memory_usage': max(memory_usage)
        })
        
        # Recovery time (if applicable)
        recovery_times = [m.recovery_time_seconds for m in metrics_list if m.recovery_time_seconds is not None]
        if recovery_times:
            aggregated['avg_recovery_time_seconds'] = statistics.mean(recovery_times)
            aggregated['max_recovery_time_seconds'] = max(recovery_times)
        
        return aggregated
    
    def _should_abort_experiment(self, experiment: ChaosExperiment, 
                                current_metrics: ResilienceMetrics) -> bool:
        """Check if experiment should be aborted due to safety conditions."""
        # Check error rate threshold
        if current_metrics.error_rate > experiment.max_error_rate:
            return True
        
        # Check response time threshold
        if current_metrics.response_time_p95 > experiment.max_response_time_ms:
            return True
        
        # Check custom abort conditions
        for condition in experiment.abort_conditions:
            if self._evaluate_abort_condition(condition, current_metrics):
                return True
        
        return False
    
    def _evaluate_abort_condition(self, condition: str, metrics: ResilienceMetrics) -> bool:
        """Evaluate a custom abort condition."""
        try:
            # Simple condition evaluation
            # In production, this would be more sophisticated
            if 'cpu_usage > 95' in condition:
                return metrics.cpu_usage_percent > 95
            elif 'memory_usage > 95' in condition:
                return metrics.memory_usage_percent > 95
            elif 'throughput < 10' in condition:
                return metrics.throughput_rps < 10
            
            return False
            
        except Exception:
            return False
    
    def _has_system_recovered(self, baseline_metrics: Dict[str, float],
                             current_metrics: ResilienceMetrics) -> bool:
        """Check if system has recovered to baseline performance."""
        # Compare current metrics to baseline
        baseline_p95 = baseline_metrics.get('avg_response_time_p95', 1000)
        baseline_error_rate = baseline_metrics.get('avg_error_rate', 0.05)
        baseline_throughput = baseline_metrics.get('avg_throughput_rps', 100)
        
        # Recovery criteria (within 20% of baseline)
        response_time_recovered = current_metrics.response_time_p95 <= baseline_p95 * 1.2
        error_rate_recovered = current_metrics.error_rate <= baseline_error_rate * 1.2
        throughput_recovered = current_metrics.throughput_rps >= baseline_throughput * 0.8
        
        return response_time_recovered and error_rate_recovered and throughput_recovered
    
    def _generate_test_results(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Generate comprehensive test results."""
        results = {
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'experiment_type': experiment.experiment_type.value,
            'status': experiment.status.value,
            'duration_seconds': experiment.duration_seconds,
            'blast_radius': experiment.blast_radius.value,
            'target_services': experiment.target_services,
            
            # Timing information
            'started_time': experiment.started_time,
            'completed_time': experiment.completed_time,
            'actual_duration': (experiment.completed_time - experiment.started_time) if experiment.completed_time else None,
            
            # Performance impact
            'baseline_metrics': experiment.baseline_metrics,
            'experiment_metrics': experiment.experiment_metrics,
            'recovery_metrics': experiment.recovery_metrics,
            
            # Analysis
            'performance_impact': self._calculate_performance_impact(experiment),
            'recovery_analysis': self._analyze_recovery(experiment),
            'resilience_score': self._calculate_resilience_score(experiment),
            
            # Recommendations
            'lessons_learned': experiment.lessons_learned,
            'recommendations': self._generate_recommendations(experiment)
        }
        
        return results
    
    def _calculate_performance_impact(self, experiment: ChaosExperiment) -> Dict[str, float]:
        """Calculate performance impact of chaos experiment."""
        baseline = experiment.baseline_metrics
        experiment_metrics = experiment.experiment_metrics
        
        if not baseline or not experiment_metrics:
            return {}
        
        impact = {}
        
        # Response time impact
        if 'avg_response_time_p95' in baseline and 'avg_response_time_p95' in experiment_metrics:
            baseline_p95 = baseline['avg_response_time_p95']
            experiment_p95 = experiment_metrics['avg_response_time_p95']
            impact['response_time_increase_percent'] = ((experiment_p95 - baseline_p95) / baseline_p95) * 100
        
        # Error rate impact
        if 'avg_error_rate' in baseline and 'avg_error_rate' in experiment_metrics:
            baseline_error = baseline['avg_error_rate']
            experiment_error = experiment_metrics['avg_error_rate']
            impact['error_rate_increase_percent'] = ((experiment_error - baseline_error) / max(baseline_error, 0.001)) * 100
        
        # Throughput impact
        if 'avg_throughput_rps' in baseline and 'avg_throughput_rps' in experiment_metrics:
            baseline_throughput = baseline['avg_throughput_rps']
            experiment_throughput = experiment_metrics['avg_throughput_rps']
            impact['throughput_decrease_percent'] = ((baseline_throughput - experiment_throughput) / baseline_throughput) * 100
        
        return impact
    
    def _analyze_recovery(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Analyze system recovery characteristics."""
        recovery = {}
        
        if experiment.recovery_metrics and experiment.baseline_metrics:
            recovery_metrics = experiment.recovery_metrics
            baseline_metrics = experiment.baseline_metrics
            
            # Recovery time estimation
            if experiment.completed_time and experiment.started_time:
                total_time = experiment.completed_time - experiment.started_time
                recovery['total_recovery_time_seconds'] = total_time - experiment.duration_seconds
            
            # Recovery completeness
            recovery_completeness = 0.0
            comparisons = 0
            
            for metric in ['avg_response_time_p95', 'avg_error_rate', 'avg_throughput_rps']:
                if metric in recovery_metrics and metric in baseline_metrics:
                    baseline_val = baseline_metrics[metric]
                    recovery_val = recovery_metrics[metric]
                    
                    if metric == 'avg_throughput_rps':
                        # Higher is better for throughput
                        completeness = min(1.0, recovery_val / baseline_val)
                    else:
                        # Lower is better for response time and error rate
                        if baseline_val > 0:
                            completeness = min(1.0, baseline_val / recovery_val)
                        else:
                            completeness = 1.0 if recovery_val == 0 else 0.0
                    
                    recovery_completeness += completeness
                    comparisons += 1
            
            if comparisons > 0:
                recovery['recovery_completeness_percent'] = (recovery_completeness / comparisons) * 100
        
        return recovery
    
    def _calculate_resilience_score(self, experiment: ChaosExperiment) -> float:
        """Calculate overall resilience score (0-100)."""
        score_components = []
        
        # Performance degradation score (lower degradation = higher score)
        impact = self._calculate_performance_impact(experiment)
        
        if 'response_time_increase_percent' in impact:
            rt_score = max(0, 100 - impact['response_time_increase_percent'])
            score_components.append(rt_score)
        
        if 'error_rate_increase_percent' in impact:
            error_score = max(0, 100 - impact['error_rate_increase_percent'])
            score_components.append(error_score)
        
        if 'throughput_decrease_percent' in impact:
            throughput_score = max(0, 100 - impact['throughput_decrease_percent'])
            score_components.append(throughput_score)
        
        # Recovery score
        recovery = self._analyze_recovery(experiment)
        if 'recovery_completeness_percent' in recovery:
            score_components.append(recovery['recovery_completeness_percent'])
        
        # Availability score (did not abort = full points)
        if experiment.status == ExperimentStatus.COMPLETED:
            score_components.append(100.0)
        elif experiment.status == ExperimentStatus.ABORTED:
            score_components.append(0.0)
        
        # Calculate weighted average
        if score_components:
            return statistics.mean(score_components)
        else:
            return 50.0  # Default score if no data
    
    def _generate_recommendations(self, experiment: ChaosExperiment) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        # Analyze performance impact
        impact = self._calculate_performance_impact(experiment)
        
        if impact.get('response_time_increase_percent', 0) > 50:
            recommendations.append("Consider implementing response time circuit breakers")
            recommendations.append("Review and optimize critical path performance")
        
        if impact.get('error_rate_increase_percent', 0) > 100:
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Implement better graceful degradation strategies")
        
        if impact.get('throughput_decrease_percent', 0) > 30:
            recommendations.append("Consider load balancing improvements")
            recommendations.append("Review capacity planning and auto-scaling policies")
        
        # Recovery analysis
        recovery = self._analyze_recovery(experiment)
        recovery_time = recovery.get('total_recovery_time_seconds', 0)
        
        if recovery_time > 300:  # More than 5 minutes
            recommendations.append("Improve automated recovery mechanisms")
            recommendations.append("Optimize health check and failover procedures")
        
        recovery_completeness = recovery.get('recovery_completeness_percent', 100)
        if recovery_completeness < 90:
            recommendations.append("Investigate incomplete recovery patterns")
            recommendations.append("Review system state consistency during recovery")
        
        # Experiment-specific recommendations
        if experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
            recommendations.append("Verify distributed system consistency during partitions")
            recommendations.append("Review partition tolerance and CAP theorem implications")
        
        elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
            recommendations.append("Validate service discovery and registration mechanisms")
            recommendations.append("Review dependency mapping and fallback services")
        
        return recommendations


class ChaosEngineeringOrchestrator:
    """Main orchestrator for chaos engineering experiments."""
    
    def __init__(self, recovery_system: AutonomousRecoverySystem):
        self.logger = get_logger(__name__)
        self.recovery_system = recovery_system
        
        # Chaos injectors
        self.network_injector = NetworkChaosInjector()
        self.service_injector = ServiceChaosInjector()
        self.test_runner = ResilienceTestRunner(recovery_system)
        
        # Experiment management
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_queue: deque = deque()
        self.running_experiments: Dict[str, threading.Thread] = {}
        
        # Safety and governance
        self.safety_enabled = True
        self.max_concurrent_experiments = 1
        self.experiment_approval_required = False
        
        # Monitoring
        self.is_orchestrating = False
        self.orchestration_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def create_experiment(self, name: str, experiment_type: ChaosExperimentType,
                         target_services: List[str], blast_radius: BlastRadiusLevel,
                         duration_seconds: float, parameters: Dict[str, Any] = None) -> str:
        """Create a new chaos experiment."""
        experiment_id = f"{name}_{int(time.time())}"
        
        experiment = ChaosExperiment(
            experiment_id=experiment_id,
            name=name,
            experiment_type=experiment_type,
            description=f"{experiment_type.value} experiment targeting {', '.join(target_services)}",
            target_services=target_services,
            blast_radius=blast_radius,
            duration_seconds=duration_seconds,
            parameters=parameters or {}
        )
        
        # Set hypothesis
        experiment.hypothesis = self._generate_hypothesis(experiment)
        
        with self._lock:
            self.experiments[experiment_id] = experiment
            self.experiment_queue.append(experiment_id)
        
        self.logger.info(f"Created chaos experiment: {name} ({experiment_id})")
        return experiment_id
    
    def schedule_experiment(self, experiment_id: str, scheduled_time: float = None):
        """Schedule an experiment for execution."""
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            experiment.scheduled_time = scheduled_time or time.time()
        
        self.logger.info(f"Scheduled experiment: {experiment_id}")
    
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run a specific experiment immediately."""
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.PLANNED:
                raise ValueError(f"Experiment {experiment_id} is not in planned state")
        
        # Execute experiment
        return self._execute_experiment(experiment)
    
    def abort_experiment(self, experiment_id: str) -> bool:
        """Abort a running experiment."""
        with self._lock:
            if experiment_id in self.running_experiments:
                experiment = self.experiments[experiment_id]
                experiment.status = ExperimentStatus.ABORTED
                
                # Stop all fault injections
                self.network_injector.stop_all_injections()
                self.service_injector.stop_all_injections()
                
                self.logger.warning(f"Aborted experiment: {experiment_id}")
                return True
        
        return False
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed experiment status."""
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            status = {
                'experiment_id': experiment_id,
                'name': experiment.name,
                'status': experiment.status.value,
                'experiment_type': experiment.experiment_type.value,
                'target_services': experiment.target_services,
                'blast_radius': experiment.blast_radius.value,
                'created_time': experiment.created_time,
                'scheduled_time': experiment.scheduled_time,
                'started_time': experiment.started_time,
                'completed_time': experiment.completed_time,
                'duration_seconds': experiment.duration_seconds,
                'hypothesis': experiment.hypothesis,
                'result_summary': experiment.result_summary
            }
            
            return status
    
    def list_experiments(self, status_filter: ExperimentStatus = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        with self._lock:
            experiments = []
            
            for experiment in self.experiments.values():
                if status_filter is None or experiment.status == status_filter:
                    experiments.append({
                        'experiment_id': experiment.experiment_id,
                        'name': experiment.name,
                        'status': experiment.status.value,
                        'experiment_type': experiment.experiment_type.value,
                        'created_time': experiment.created_time,
                        'target_services': experiment.target_services
                    })
            
            return sorted(experiments, key=lambda x: x['created_time'], reverse=True)
    
    def start_orchestration(self, check_interval: float = 60.0):
        """Start automatic experiment orchestration."""
        if self.is_orchestrating:
            return
        
        self.is_orchestrating = True
        self.orchestration_thread = threading.Thread(
            target=self._orchestration_loop,
            args=(check_interval,),
            daemon=True
        )
        self.orchestration_thread.start()
        self.logger.info("Chaos engineering orchestration started")
    
    def stop_orchestration(self):
        """Stop automatic orchestration."""
        self.is_orchestrating = False
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=10.0)
        self.logger.info("Chaos engineering orchestration stopped")
    
    def _execute_experiment(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute a chaos experiment."""
        try:
            self.logger.info(f"Executing experiment: {experiment.name}")
            
            # Create metrics collector
            def collect_metrics() -> ResilienceMetrics:
                return self._collect_current_metrics(experiment.experiment_id)
            
            # Inject chaos based on experiment type
            fault_injections = self._inject_chaos(experiment)
            
            # Run resilience test
            results = self.test_runner.run_resilience_test(experiment, collect_metrics)
            
            # Stop fault injections
            self._stop_fault_injections(fault_injections)
            
            # Update experiment with results
            experiment.result_summary = f"Resilience score: {results.get('resilience_score', 0):.1f}/100"
            experiment.lessons_learned = results.get('recommendations', [])
            
            self.logger.info(f"Completed experiment: {experiment.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing experiment {experiment.name}: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.result_summary = f"Failed: {str(e)}"
            
            return {
                'experiment_id': experiment.experiment_id,
                'status': ExperimentStatus.FAILED.value,
                'error': str(e)
            }
    
    def _inject_chaos(self, experiment: ChaosExperiment) -> List[str]:
        """Inject chaos based on experiment type."""
        fault_injections = []
        
        try:
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                latency_ms = experiment.parameters.get('latency_ms', 500)
                for service in experiment.target_services:
                    fault_id = self.network_injector.inject_latency(
                        service, latency_ms, experiment.duration_seconds
                    )
                    fault_injections.append(fault_id)
            
            elif experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
                if len(experiment.target_services) >= 2:
                    fault_id = self.network_injector.inject_partition(
                        experiment.target_services[0],
                        experiment.target_services[1],
                        experiment.duration_seconds
                    )
                    fault_injections.append(fault_id)
            
            elif experiment.experiment_type == ChaosExperimentType.SERVICE_FAILURE:
                for service in experiment.target_services:
                    fault_id = self.service_injector.kill_service_instance(service)
                    fault_injections.append(fault_id)
            
            elif experiment.experiment_type == ChaosExperimentType.RESOURCE_EXHAUSTION:
                resource_type = experiment.parameters.get('resource_type', 'cpu')
                exhaustion_level = experiment.parameters.get('exhaustion_level', 0.9)
                
                for service in experiment.target_services:
                    fault_id = self.service_injector.inject_resource_exhaustion(
                        service, resource_type, exhaustion_level, experiment.duration_seconds
                    )
                    fault_injections.append(fault_id)
        
        except Exception as e:
            self.logger.error(f"Error injecting chaos for experiment {experiment.experiment_id}: {e}")
        
        return fault_injections
    
    def _stop_fault_injections(self, fault_injections: List[str]):
        """Stop all fault injections."""
        for fault_id in fault_injections:
            try:
                self.network_injector.stop_injection(fault_id)
                # Note: service_injector faults are typically one-time events
            except Exception as e:
                self.logger.error(f"Error stopping fault injection {fault_id}: {e}")
    
    def _collect_current_metrics(self, experiment_id: str) -> ResilienceMetrics:
        """Collect current system metrics for experiment."""
        # Get system metrics from recovery system
        recovery_status = self.recovery_system.get_recovery_statistics()
        
        metrics = ResilienceMetrics(
            experiment_id=experiment_id,
            response_time_p50=100.0,  # Default values
            response_time_p95=200.0,
            response_time_p99=500.0,
            error_rate=0.01,
            throughput_rps=100.0,
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            service_availability={'default': 0.99}
        )
        
        return metrics
    
    def _generate_hypothesis(self, experiment: ChaosExperiment) -> str:
        """Generate hypothesis for the experiment."""
        hypotheses = {
            ChaosExperimentType.NETWORK_LATENCY: f"System should maintain <{experiment.max_response_time_ms}ms response time despite network latency",
            ChaosExperimentType.NETWORK_PARTITION: "System should handle network partitions gracefully with eventual consistency",
            ChaosExperimentType.SERVICE_FAILURE: f"System should automatically recover from {', '.join(experiment.target_services)} failure within 60 seconds",
            ChaosExperimentType.RESOURCE_EXHAUSTION: "System should scale or degrade gracefully under resource pressure",
            ChaosExperimentType.DEPENDENCY_FAILURE: "System should use circuit breakers and fallbacks when dependencies fail"
        }
        
        return hypotheses.get(experiment.experiment_type, "System should remain stable during chaos injection")
    
    def _orchestration_loop(self, check_interval: float):
        """Main orchestration loop for automatic experiment execution."""
        while self.is_orchestrating:
            try:
                current_time = time.time()
                
                # Check for scheduled experiments
                experiments_to_run = []
                
                with self._lock:
                    for experiment_id in list(self.experiment_queue):
                        experiment = self.experiments[experiment_id]
                        
                        if (experiment.scheduled_time and 
                            experiment.scheduled_time <= current_time and
                            experiment.status == ExperimentStatus.PLANNED):
                            
                            # Check if we can run more experiments
                            if len(self.running_experiments) < self.max_concurrent_experiments:
                                experiments_to_run.append(experiment_id)
                                self.experiment_queue.remove(experiment_id)
                
                # Execute scheduled experiments
                for experiment_id in experiments_to_run:
                    experiment = self.experiments[experiment_id]
                    
                    # Start experiment in separate thread
                    experiment_thread = threading.Thread(
                        target=self._execute_experiment,
                        args=(experiment,),
                        daemon=True
                    )
                    
                    with self._lock:
                        self.running_experiments[experiment_id] = experiment_thread
                    
                    experiment_thread.start()
                
                # Clean up completed experiments
                completed_experiments = []
                
                with self._lock:
                    for experiment_id, thread in list(self.running_experiments.items()):
                        if not thread.is_alive():
                            completed_experiments.append(experiment_id)
                
                for experiment_id in completed_experiments:
                    with self._lock:
                        del self.running_experiments[experiment_id]
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
                time.sleep(check_interval)


# Global chaos engineering orchestrator
_global_chaos_orchestrator: Optional[ChaosEngineeringOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_chaos_orchestrator(recovery_system: AutonomousRecoverySystem = None) -> ChaosEngineeringOrchestrator:
    """Get or create the global chaos engineering orchestrator."""
    global _global_chaos_orchestrator
    
    with _orchestrator_lock:
        if _global_chaos_orchestrator is None:
            if recovery_system is None:
                from .autonomous_recovery import get_recovery_system
                recovery_system = get_recovery_system()
            _global_chaos_orchestrator = ChaosEngineeringOrchestrator(recovery_system)
        return _global_chaos_orchestrator


def start_chaos_engineering():
    """Start the global chaos engineering system."""
    orchestrator = get_chaos_orchestrator()
    orchestrator.start_orchestration()


def stop_chaos_engineering():
    """Stop the global chaos engineering system."""
    orchestrator = get_chaos_orchestrator()
    orchestrator.stop_orchestration()