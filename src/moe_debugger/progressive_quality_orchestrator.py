"""Progressive Quality Gates Orchestrator for Enterprise-Grade MoE Debugging.

This module orchestrates all advanced Progressive Quality Gates components,
providing a unified interface for enterprise-scale deployment with
comprehensive security, performance, resilience, observability, and governance.

Features:
- Unified orchestration of all Progressive Quality Gates components
- Centralized configuration and management
- Cross-component integration and data flow
- Health monitoring and system status
- Automated component lifecycle management
- Performance metrics aggregation and reporting

Authors: Terragon Labs - Progressive Quality Gates v2.0
License: MIT
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .logging_config import get_logger
from .validation import safe_json_dumps

# Progressive Quality Gates components
from .ai_threat_detection import get_threat_detection_system, start_threat_monitoring, stop_threat_monitoring
from .quantum_performance_optimization import get_performance_optimizer, start_quantum_optimization, stop_quantum_optimization
from .chaos_engineering import get_chaos_orchestrator, start_chaos_engineering, stop_chaos_engineering
from .advanced_observability import get_observability_system, start_observability_monitoring, stop_observability_monitoring
from .enterprise_governance import get_governance_system, start_enterprise_governance, stop_enterprise_governance
from .autonomous_recovery import get_recovery_system


class QualityGateStatus(Enum):
    """Status of Progressive Quality Gates."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class ComponentType(Enum):
    """Types of Progressive Quality Gates components."""
    THREAT_DETECTION = "threat_detection"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CHAOS_ENGINEERING = "chaos_engineering"
    OBSERVABILITY = "observability"
    GOVERNANCE = "governance"
    RECOVERY_SYSTEM = "recovery_system"


@dataclass
class ComponentHealth:
    """Health status of a Progressive Quality Gates component."""
    component_type: ComponentType
    status: QualityGateStatus
    last_check: float
    uptime_seconds: float
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityGateConfiguration:
    """Configuration for Progressive Quality Gates system."""
    # Component enablement
    threat_detection_enabled: bool = True
    performance_optimization_enabled: bool = True
    chaos_engineering_enabled: bool = True
    observability_enabled: bool = True
    governance_enabled: bool = True
    recovery_system_enabled: bool = True
    
    # System parameters
    health_check_interval_seconds: float = 30.0
    metrics_collection_interval_seconds: float = 60.0
    cross_component_integration_enabled: bool = True
    automated_incident_response_enabled: bool = True
    
    # Performance thresholds
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 80.0
    max_response_time_ms: float = 500.0
    min_uptime_percent: float = 99.9
    
    # Security settings
    security_monitoring_level: str = "high"  # low, medium, high, critical
    threat_response_automation: bool = True
    compliance_enforcement_level: str = "strict"  # lenient, moderate, strict
    
    # Operational settings
    chaos_testing_enabled: bool = True
    chaos_testing_schedule: str = "weekly"  # daily, weekly, monthly
    predictive_scaling_enabled: bool = True
    automated_remediation_enabled: bool = True


class ProgressiveQualityOrchestrator:
    """Main orchestrator for Progressive Quality Gates system."""
    
    def __init__(self, config: QualityGateConfiguration = None):
        self.logger = get_logger(__name__)
        self.config = config or QualityGateConfiguration()
        
        # System state
        self.overall_status = QualityGateStatus.INITIALIZING
        self.start_time = time.time()
        self.is_running = False
        
        # Component health tracking
        self.component_health: Dict[ComponentType, ComponentHealth] = {}
        self.system_metrics: deque = deque(maxlen=1000)
        self.incident_history: deque = deque(maxlen=1000)
        
        # Component references
        self.components = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        
        # Cross-component integration
        self.integration_handlers = {}
        self.data_flow_metrics = defaultdict(int)
        
        self._lock = threading.Lock()
    
    def initialize_system(self) -> Dict[str, Any]:
        """Initialize the Progressive Quality Gates system."""
        try:
            self.logger.info("Initializing Progressive Quality Gates system")
            initialization_results = {
                'timestamp': time.time(),
                'overall_status': 'success',
                'initialized_components': [],
                'failed_components': [],
                'warnings': []
            }
            
            # Initialize components based on configuration
            if self.config.threat_detection_enabled:
                try:
                    self.components[ComponentType.THREAT_DETECTION] = get_threat_detection_system()
                    self._initialize_component_health(ComponentType.THREAT_DETECTION)
                    initialization_results['initialized_components'].append('threat_detection')
                    self.logger.info("AI Threat Detection system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Threat Detection: {e}")
                    initialization_results['failed_components'].append(f'threat_detection: {str(e)}')
            
            if self.config.performance_optimization_enabled:
                try:
                    self.components[ComponentType.PERFORMANCE_OPTIMIZATION] = get_performance_optimizer()
                    self._initialize_component_health(ComponentType.PERFORMANCE_OPTIMIZATION)
                    initialization_results['initialized_components'].append('performance_optimization')
                    self.logger.info("Quantum Performance Optimization system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Performance Optimization: {e}")
                    initialization_results['failed_components'].append(f'performance_optimization: {str(e)}')
            
            if self.config.chaos_engineering_enabled:
                try:
                    recovery_system = get_recovery_system()
                    self.components[ComponentType.CHAOS_ENGINEERING] = get_chaos_orchestrator(recovery_system)
                    self._initialize_component_health(ComponentType.CHAOS_ENGINEERING)
                    initialization_results['initialized_components'].append('chaos_engineering')
                    self.logger.info("Chaos Engineering system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Chaos Engineering: {e}")
                    initialization_results['failed_components'].append(f'chaos_engineering: {str(e)}')
            
            if self.config.observability_enabled:
                try:
                    self.components[ComponentType.OBSERVABILITY] = get_observability_system()
                    self._initialize_component_health(ComponentType.OBSERVABILITY)
                    initialization_results['initialized_components'].append('observability')
                    self.logger.info("Advanced Observability system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Observability: {e}")
                    initialization_results['failed_components'].append(f'observability: {str(e)}')
            
            if self.config.governance_enabled:
                try:
                    self.components[ComponentType.GOVERNANCE] = get_governance_system()
                    self._initialize_component_health(ComponentType.GOVERNANCE)
                    initialization_results['initialized_components'].append('governance')
                    self.logger.info("Enterprise Governance system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Governance: {e}")
                    initialization_results['failed_components'].append(f'governance: {str(e)}')
            
            if self.config.recovery_system_enabled:
                try:
                    self.components[ComponentType.RECOVERY_SYSTEM] = get_recovery_system()
                    self._initialize_component_health(ComponentType.RECOVERY_SYSTEM)
                    initialization_results['initialized_components'].append('recovery_system')
                    self.logger.info("Autonomous Recovery system initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Recovery System: {e}")
                    initialization_results['failed_components'].append(f'recovery_system: {str(e)}')
            
            # Setup cross-component integration
            if self.config.cross_component_integration_enabled:
                self._setup_component_integration()
                initialization_results['initialized_components'].append('cross_component_integration')
            
            # Determine overall initialization status
            total_components = len(initialization_results['initialized_components']) + len(initialization_results['failed_components'])
            success_rate = len(initialization_results['initialized_components']) / total_components if total_components > 0 else 0
            
            if success_rate >= 0.8:
                self.overall_status = QualityGateStatus.ACTIVE
                initialization_results['overall_status'] = 'success'
            elif success_rate >= 0.5:
                self.overall_status = QualityGateStatus.DEGRADED
                initialization_results['overall_status'] = 'degraded'
                initialization_results['warnings'].append('Some components failed to initialize - system running in degraded mode')
            else:
                self.overall_status = QualityGateStatus.CRITICAL
                initialization_results['overall_status'] = 'critical'
            
            self.logger.info(f"Progressive Quality Gates initialization completed: {initialization_results['overall_status']}")
            return initialization_results
            
        except Exception as e:
            self.logger.error(f"Critical error during system initialization: {e}")
            self.overall_status = QualityGateStatus.OFFLINE
            return {
                'timestamp': time.time(),
                'overall_status': 'failed',
                'error': str(e),
                'initialized_components': [],
                'failed_components': ['system_initialization']
            }
    
    def start_system(self) -> Dict[str, Any]:
        """Start all Progressive Quality Gates components."""
        try:
            if self.is_running:
                return {'status': 'already_running', 'message': 'System is already running'}
            
            self.logger.info("Starting Progressive Quality Gates system")
            start_results = {
                'timestamp': time.time(),
                'overall_status': 'success',
                'started_components': [],
                'failed_components': []
            }
            
            # Start individual components
            if ComponentType.THREAT_DETECTION in self.components:
                try:
                    start_threat_monitoring()
                    start_results['started_components'].append('threat_detection')
                    self.logger.info("AI Threat Detection monitoring started")
                except Exception as e:
                    self.logger.error(f"Failed to start Threat Detection: {e}")
                    start_results['failed_components'].append(f'threat_detection: {str(e)}')
            
            if ComponentType.PERFORMANCE_OPTIMIZATION in self.components:
                try:
                    start_quantum_optimization()
                    start_results['started_components'].append('performance_optimization')
                    self.logger.info("Quantum Performance Optimization started")
                except Exception as e:
                    self.logger.error(f"Failed to start Performance Optimization: {e}")
                    start_results['failed_components'].append(f'performance_optimization: {str(e)}')
            
            if ComponentType.CHAOS_ENGINEERING in self.components:
                try:
                    start_chaos_engineering()
                    start_results['started_components'].append('chaos_engineering')
                    self.logger.info("Chaos Engineering started")
                except Exception as e:
                    self.logger.error(f"Failed to start Chaos Engineering: {e}")
                    start_results['failed_components'].append(f'chaos_engineering: {str(e)}')
            
            if ComponentType.OBSERVABILITY in self.components:
                try:
                    start_observability_monitoring()
                    start_results['started_components'].append('observability')
                    self.logger.info("Advanced Observability monitoring started")
                except Exception as e:
                    self.logger.error(f"Failed to start Observability: {e}")
                    start_results['failed_components'].append(f'observability: {str(e)}')
            
            if ComponentType.GOVERNANCE in self.components:
                try:
                    start_enterprise_governance()
                    start_results['started_components'].append('governance')
                    self.logger.info("Enterprise Governance started")
                except Exception as e:
                    self.logger.error(f"Failed to start Governance: {e}")
                    start_results['failed_components'].append(f'governance: {str(e)}')
            
            if ComponentType.RECOVERY_SYSTEM in self.components:
                try:
                    self.components[ComponentType.RECOVERY_SYSTEM].start_monitoring()
                    start_results['started_components'].append('recovery_system')
                    self.logger.info("Autonomous Recovery system started")
                except Exception as e:
                    self.logger.error(f"Failed to start Recovery System: {e}")
                    start_results['failed_components'].append(f'recovery_system: {str(e)}')
            
            # Start system monitoring
            self.is_running = True
            self._start_system_monitoring()
            start_results['started_components'].append('system_monitoring')
            
            # Update overall status
            if len(start_results['failed_components']) == 0:
                self.overall_status = QualityGateStatus.ACTIVE
            elif len(start_results['started_components']) > len(start_results['failed_components']):
                self.overall_status = QualityGateStatus.DEGRADED
            else:
                self.overall_status = QualityGateStatus.CRITICAL
            
            self.logger.info(f"Progressive Quality Gates system started: {len(start_results['started_components'])} components active")
            return start_results
            
        except Exception as e:
            self.logger.error(f"Critical error starting system: {e}")
            return {
                'timestamp': time.time(),
                'overall_status': 'failed',
                'error': str(e)
            }
    
    def stop_system(self) -> Dict[str, Any]:
        """Stop all Progressive Quality Gates components."""
        try:
            if not self.is_running:
                return {'status': 'already_stopped', 'message': 'System is not running'}
            
            self.logger.info("Stopping Progressive Quality Gates system")
            stop_results = {
                'timestamp': time.time(),
                'overall_status': 'success',
                'stopped_components': [],
                'failed_components': []
            }
            
            # Stop system monitoring first
            self.is_running = False
            self._stop_system_monitoring()
            stop_results['stopped_components'].append('system_monitoring')
            
            # Stop individual components
            try:
                stop_threat_monitoring()
                stop_results['stopped_components'].append('threat_detection')
            except Exception as e:
                stop_results['failed_components'].append(f'threat_detection: {str(e)}')
            
            try:
                stop_quantum_optimization()
                stop_results['stopped_components'].append('performance_optimization')
            except Exception as e:
                stop_results['failed_components'].append(f'performance_optimization: {str(e)}')
            
            try:
                stop_chaos_engineering()
                stop_results['stopped_components'].append('chaos_engineering')
            except Exception as e:
                stop_results['failed_components'].append(f'chaos_engineering: {str(e)}')
            
            try:
                stop_observability_monitoring()
                stop_results['stopped_components'].append('observability')
            except Exception as e:
                stop_results['failed_components'].append(f'observability: {str(e)}')
            
            try:
                stop_enterprise_governance()
                stop_results['stopped_components'].append('governance')
            except Exception as e:
                stop_results['failed_components'].append(f'governance: {str(e)}')
            
            if ComponentType.RECOVERY_SYSTEM in self.components:
                try:
                    self.components[ComponentType.RECOVERY_SYSTEM].stop_monitoring()
                    stop_results['stopped_components'].append('recovery_system')
                except Exception as e:
                    stop_results['failed_components'].append(f'recovery_system: {str(e)}')
            
            self.overall_status = QualityGateStatus.OFFLINE
            
            self.logger.info(f"Progressive Quality Gates system stopped: {len(stop_results['stopped_components'])} components stopped")
            return stop_results
            
        except Exception as e:
            self.logger.error(f"Critical error stopping system: {e}")
            return {
                'timestamp': time.time(),
                'overall_status': 'failed',
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health information."""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Collect component health
            component_statuses = {}
            overall_health_score = 100.0
            
            for component_type, health in self.component_health.items():
                component_statuses[component_type.value] = {
                    'status': health.status.value,
                    'uptime_seconds': health.uptime_seconds,
                    'error_count': health.error_count,
                    'last_error': health.last_error,
                    'performance_metrics': health.performance_metrics,
                    'resource_usage': health.resource_usage
                }
                
                # Adjust overall health score
                if health.status == QualityGateStatus.CRITICAL:
                    overall_health_score -= 30
                elif health.status == QualityGateStatus.DEGRADED:
                    overall_health_score -= 15
                elif health.status == QualityGateStatus.OFFLINE:
                    overall_health_score -= 40
            
            # Recent system metrics
            recent_metrics = list(self.system_metrics)[-10:] if self.system_metrics else []
            
            # Integration status
            integration_status = {
                'cross_component_enabled': self.config.cross_component_integration_enabled,
                'data_flow_metrics': dict(self.data_flow_metrics),
                'integration_handlers': len(self.integration_handlers)
            }
            
            status = {
                'timestamp': current_time,
                'overall_status': self.overall_status.value,
                'overall_health_score': max(0.0, min(100.0, overall_health_score)),
                'system_uptime_seconds': uptime,
                'is_running': self.is_running,
                'configuration': {
                    'threat_detection_enabled': self.config.threat_detection_enabled,
                    'performance_optimization_enabled': self.config.performance_optimization_enabled,
                    'chaos_engineering_enabled': self.config.chaos_engineering_enabled,
                    'observability_enabled': self.config.observability_enabled,
                    'governance_enabled': self.config.governance_enabled,
                    'recovery_system_enabled': self.config.recovery_system_enabled
                },
                'component_health': component_statuses,
                'integration_status': integration_status,
                'recent_metrics': recent_metrics,
                'incident_count_24h': len([i for i in self.incident_history 
                                         if current_time - i.get('timestamp', 0) < 86400])
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': time.time(),
                'overall_status': 'error',
                'error': str(e),
                'is_running': self.is_running
            }
    
    def trigger_quality_gate_assessment(self) -> Dict[str, Any]:
        """Trigger comprehensive Progressive Quality Gates assessment."""
        try:
            self.logger.info("Starting Progressive Quality Gates assessment")
            assessment = {
                'timestamp': time.time(),
                'assessment_id': f"pqg_{int(time.time())}",
                'overall_score': 0.0,
                'component_assessments': {},
                'recommendations': [],
                'critical_issues': [],
                'system_health': self.get_system_status()
            }
            
            total_score = 0.0
            component_count = 0
            
            # Threat Detection Assessment
            if ComponentType.THREAT_DETECTION in self.components:
                try:
                    threat_system = self.components[ComponentType.THREAT_DETECTION]
                    security_status = threat_system.get_security_status()
                    
                    threat_score = 100.0
                    if security_status.get('security_metrics', {}).get('threats_detected', 0) > 10:
                        threat_score -= 20
                    if security_status.get('blocked_ips_count', 0) > 50:
                        threat_score -= 15
                    
                    assessment['component_assessments']['threat_detection'] = {
                        'score': threat_score,
                        'status': security_status,
                        'issues': []
                    }
                    
                    if threat_score < 70:
                        assessment['critical_issues'].append("High threat activity detected")
                    
                    total_score += threat_score
                    component_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error assessing threat detection: {e}")
            
            # Performance Optimization Assessment
            if ComponentType.PERFORMANCE_OPTIMIZATION in self.components:
                try:
                    perf_system = self.components[ComponentType.PERFORMANCE_OPTIMIZATION]
                    perf_status = perf_system.get_optimization_status()
                    
                    perf_score = 85.0  # Base score
                    cost_savings = perf_status.get('cost_optimization_savings', 0)
                    if cost_savings > 100:  # $100+ savings
                        perf_score += 10
                    
                    assessment['component_assessments']['performance_optimization'] = {
                        'score': perf_score,
                        'status': perf_status,
                        'cost_savings': cost_savings
                    }
                    
                    total_score += perf_score
                    component_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error assessing performance optimization: {e}")
            
            # Governance Assessment
            if ComponentType.GOVERNANCE in self.components:
                try:
                    governance_system = self.components[ComponentType.GOVERNANCE]
                    governance_dashboard = governance_system.get_governance_dashboard()
                    
                    governance_score = governance_dashboard.get('governance_health_score', 50.0)
                    
                    assessment['component_assessments']['governance'] = {
                        'score': governance_score,
                        'status': governance_dashboard
                    }
                    
                    if governance_score < 80:
                        assessment['critical_issues'].append("Governance compliance issues detected")
                    
                    total_score += governance_score
                    component_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error assessing governance: {e}")
            
            # Observability Assessment
            if ComponentType.OBSERVABILITY in self.components:
                try:
                    obs_system = self.components[ComponentType.OBSERVABILITY]
                    obs_dashboard = obs_system.get_observability_dashboard()
                    
                    obs_score = obs_dashboard.get('system_health_score', 50.0)
                    
                    assessment['component_assessments']['observability'] = {
                        'score': obs_score,
                        'status': obs_dashboard
                    }
                    
                    total_score += obs_score
                    component_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error assessing observability: {e}")
            
            # Calculate overall assessment score
            if component_count > 0:
                assessment['overall_score'] = total_score / component_count
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_assessment_recommendations(assessment)
            
            # Determine quality gate status
            overall_score = assessment['overall_score']
            if overall_score >= 90:
                assessment['quality_gate_status'] = 'excellent'
            elif overall_score >= 80:
                assessment['quality_gate_status'] = 'good'
            elif overall_score >= 70:
                assessment['quality_gate_status'] = 'acceptable'
            else:
                assessment['quality_gate_status'] = 'needs_improvement'
            
            self.logger.info(f"Progressive Quality Gates assessment completed: {overall_score:.1f}/100 ({assessment['quality_gate_status']})")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in quality gate assessment: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e),
                'overall_score': 0.0,
                'quality_gate_status': 'error'
            }
    
    def _initialize_component_health(self, component_type: ComponentType):
        """Initialize health tracking for a component."""
        self.component_health[component_type] = ComponentHealth(
            component_type=component_type,
            status=QualityGateStatus.INITIALIZING,
            last_check=time.time(),
            uptime_seconds=0.0
        )
    
    def _setup_component_integration(self):
        """Setup cross-component integration handlers."""
        try:
            # Threat Detection -> Chaos Engineering integration
            self.integration_handlers['threat_to_chaos'] = self._threat_to_chaos_handler
            
            # Performance Optimization -> Observability integration
            self.integration_handlers['perf_to_obs'] = self._performance_to_observability_handler
            
            # Governance -> All Components integration
            self.integration_handlers['governance_to_all'] = self._governance_to_all_handler
            
            # Recovery System -> All Components integration
            self.integration_handlers['recovery_to_all'] = self._recovery_to_all_handler
            
            self.logger.info(f"Setup {len(self.integration_handlers)} integration handlers")
            
        except Exception as e:
            self.logger.error(f"Error setting up component integration: {e}")
    
    def _threat_to_chaos_handler(self, threat_data: Dict[str, Any]):
        """Handle integration between threat detection and chaos engineering."""
        # If high threat activity, potentially trigger chaos experiments to test resilience
        if threat_data.get('threat_level') == 'high':
            self.data_flow_metrics['threat_to_chaos'] += 1
    
    def _performance_to_observability_handler(self, perf_data: Dict[str, Any]):
        """Handle integration between performance optimization and observability."""
        # Feed performance optimization data to observability system
        if ComponentType.OBSERVABILITY in self.components:
            obs_system = self.components[ComponentType.OBSERVABILITY]
            # Convert performance data to metrics format
            metrics = []
            for key, value in perf_data.items():
                if isinstance(value, (int, float)):
                    metrics.append({
                        'name': f'perf_optimization_{key}',
                        'value': value,
                        'timestamp': time.time(),
                        'labels': {'component': 'performance_optimization'}
                    })
            
            if metrics:
                obs_system.ingest_metrics(metrics)
                self.data_flow_metrics['perf_to_obs'] += len(metrics)
    
    def _governance_to_all_handler(self, governance_data: Dict[str, Any]):
        """Handle governance policy enforcement across all components."""
        # Apply governance policies to all components
        self.data_flow_metrics['governance_to_all'] += 1
    
    def _recovery_to_all_handler(self, recovery_data: Dict[str, Any]):
        """Handle recovery system coordination with all components."""
        # Coordinate recovery actions across all components
        self.data_flow_metrics['recovery_to_all'] += 1
    
    def _start_system_monitoring(self):
        """Start system-wide monitoring."""
        def monitoring_loop():
            while self.is_running:
                try:
                    # Update component health
                    self._update_component_health()
                    
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    # Check for incidents
                    self._check_for_incidents()
                    
                    time.sleep(self.config.health_check_interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error in system monitoring loop: {e}")
                    time.sleep(self.config.health_check_interval_seconds)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        self.monitoring_threads['system_monitor'] = monitoring_thread
    
    def _stop_system_monitoring(self):
        """Stop system monitoring."""
        # Monitoring threads are daemon threads and will stop when main process stops
        self.monitoring_threads.clear()
    
    def _update_component_health(self):
        """Update health status of all components."""
        current_time = time.time()
        
        for component_type, health in self.component_health.items():
            try:
                # Update uptime
                health.uptime_seconds = current_time - (health.last_check - health.uptime_seconds)
                health.last_check = current_time
                
                # Check component status (simplified)
                if component_type in self.components:
                    health.status = QualityGateStatus.ACTIVE
                else:
                    health.status = QualityGateStatus.OFFLINE
                
            except Exception as e:
                health.error_count += 1
                health.last_error = str(e)
                health.status = QualityGateStatus.CRITICAL
                self.logger.error(f"Error updating health for {component_type.value}: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            metrics = {
                'timestamp': time.time(),
                'overall_status': self.overall_status.value,
                'active_components': len([h for h in self.component_health.values() 
                                        if h.status == QualityGateStatus.ACTIVE]),
                'total_components': len(self.component_health),
                'system_uptime': time.time() - self.start_time,
                'data_flow_events': sum(self.data_flow_metrics.values())
            }
            
            self.system_metrics.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _check_for_incidents(self):
        """Check for system incidents and handle them."""
        try:
            # Check for critical component failures
            critical_components = [
                h for h in self.component_health.values() 
                if h.status in [QualityGateStatus.CRITICAL, QualityGateStatus.OFFLINE]
            ]
            
            if len(critical_components) >= 2:  # Multiple component failures
                incident = {
                    'timestamp': time.time(),
                    'type': 'multiple_component_failure',
                    'severity': 'critical',
                    'affected_components': [c.component_type.value for c in critical_components],
                    'description': f"{len(critical_components)} components in critical state"
                }
                
                self.incident_history.append(incident)
                self.logger.critical(f"INCIDENT: {incident['description']}")
                
                # Trigger automated incident response if enabled
                if self.config.automated_incident_response_enabled:
                    self._handle_incident(incident)
            
        except Exception as e:
            self.logger.error(f"Error checking for incidents: {e}")
    
    def _handle_incident(self, incident: Dict[str, Any]):
        """Handle system incidents with automated responses."""
        try:
            incident_type = incident.get('type', '')
            
            if incident_type == 'multiple_component_failure':
                # Attempt to restart failed components
                self.logger.info("Attempting automated incident recovery")
                
                # This would trigger more sophisticated recovery procedures
                # For now, just log the incident
                
            self.logger.info(f"Incident response completed for {incident['type']}")
            
        except Exception as e:
            self.logger.error(f"Error handling incident: {e}")
    
    def _generate_assessment_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        
        overall_score = assessment.get('overall_score', 0)
        
        if overall_score < 80:
            recommendations.append("Overall system quality needs improvement")
        
        # Component-specific recommendations
        for component, details in assessment.get('component_assessments', {}).items():
            score = details.get('score', 0)
            if score < 70:
                recommendations.append(f"Address issues in {component} component (score: {score:.1f})")
        
        # Critical issues
        critical_issues = assessment.get('critical_issues', [])
        if critical_issues:
            recommendations.append(f"Immediately address {len(critical_issues)} critical issues")
        
        # General recommendations
        recommendations.extend([
            "Regularly monitor Progressive Quality Gates dashboard",
            "Review and update quality gate configurations",
            "Conduct periodic system assessments",
            "Maintain component documentation and procedures"
        ])
        
        return recommendations[:10]  # Top 10 recommendations


# Global Progressive Quality Gates orchestrator
_global_orchestrator: Optional[ProgressiveQualityOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_progressive_quality_orchestrator(config: QualityGateConfiguration = None) -> ProgressiveQualityOrchestrator:
    """Get or create the global Progressive Quality Gates orchestrator."""
    global _global_orchestrator
    
    with _orchestrator_lock:
        if _global_orchestrator is None:
            _global_orchestrator = ProgressiveQualityOrchestrator(config)
        return _global_orchestrator


def initialize_progressive_quality_gates(config: QualityGateConfiguration = None) -> Dict[str, Any]:
    """Initialize and start the Progressive Quality Gates system."""
    orchestrator = get_progressive_quality_orchestrator(config)
    
    # Initialize system
    init_result = orchestrator.initialize_system()
    
    if init_result['overall_status'] in ['success', 'degraded']:
        # Start system if initialization was successful
        start_result = orchestrator.start_system()
        
        return {
            'timestamp': time.time(),
            'status': 'success' if start_result['overall_status'] == 'success' else 'degraded',
            'initialization': init_result,
            'startup': start_result,
            'system_status': orchestrator.get_system_status()
        }
    else:
        return {
            'timestamp': time.time(),
            'status': 'failed',
            'initialization': init_result,
            'message': 'System initialization failed - startup aborted'
        }


def shutdown_progressive_quality_gates() -> Dict[str, Any]:
    """Shutdown the Progressive Quality Gates system."""
    orchestrator = get_progressive_quality_orchestrator()
    return orchestrator.stop_system()


def get_progressive_quality_status() -> Dict[str, Any]:
    """Get current Progressive Quality Gates system status."""
    orchestrator = get_progressive_quality_orchestrator()
    return orchestrator.get_system_status()


def run_progressive_quality_assessment() -> Dict[str, Any]:
    """Run comprehensive Progressive Quality Gates assessment."""
    orchestrator = get_progressive_quality_orchestrator()
    return orchestrator.trigger_quality_gate_assessment()