"""Explainable-MoE-Debugger: Chrome DevTools-style debugging for Mixture of Experts models.

This package provides real-time visualization and analysis tools for understanding
expert routing, load balancing, and token attribution in MoE architectures.
"""

from .__about__ import __version__

# Conditional imports to handle missing dependencies gracefully
__all__ = ["__version__"]

try:
    from .debugger import MoEDebugger
    __all__.append("MoEDebugger")
except ImportError:
    pass

# Always include factory for compatibility
try:
    from .debugger_factory import MoEDebuggerFactory
    __all__.append("MoEDebuggerFactory")
except ImportError:
    pass

try:
    from .analyzer import MoEAnalyzer  
    __all__.append("MoEAnalyzer")
except ImportError:
    pass

try:
    from .profiler import MoEProfiler
    __all__.append("MoEProfiler")
except ImportError:
    pass

try:
    from .server import DebugServer
    __all__.append("DebugServer")
except ImportError:
    pass

# Research extensions
try:
    from .adaptive_routing import (
        AdaptiveRoutingSystem, AdaptiveRoutingConfig,
        EntropyGuidedAdaptiveRouter, DeadExpertResurrectionFramework,
        PredictiveLoadBalancer, MultiObjectiveRoutingOptimizer
    )
    __all__.extend([
        "AdaptiveRoutingSystem", "AdaptiveRoutingConfig",
        "EntropyGuidedAdaptiveRouter", "DeadExpertResurrectionFramework", 
        "PredictiveLoadBalancer", "MultiObjectiveRoutingOptimizer"
    ])
except ImportError:
    pass

try:
    from .research_validation import (
        run_comprehensive_research_validation,
        ExperimentRunner, StatisticalAnalyzer, ResearchReportGenerator
    )
    __all__.extend([
        "run_comprehensive_research_validation",
        "ExperimentRunner", "StatisticalAnalyzer", "ResearchReportGenerator"
    ])
except ImportError:
    pass

# Enhanced debugger with research integration
try:
    from .enhanced_debugger import (
        EnhancedMoEDebugger, EnhancedDebuggerConfig,
        create_enhanced_debugger
    )
    __all__.extend([
        "EnhancedMoEDebugger", "EnhancedDebuggerConfig",
        "create_enhanced_debugger"
    ])
except ImportError:
    pass

# Autonomous recovery system
try:
    from .autonomous_recovery import (
        AutonomousRecoverySystem, HealthStatus, FailurePattern,
        get_recovery_system, autonomous_recovery
    )
    __all__.extend([
        "AutonomousRecoverySystem", "HealthStatus", "FailurePattern",
        "get_recovery_system", "autonomous_recovery"
    ])
except ImportError:
    pass

# Quantum routing algorithms
try:
    from .quantum_routing import (
        QuantumRoutingSystem, QuantumSuperpositionRouter,
        QuantumAnnealingOptimizer, get_quantum_router,
        quantum_route_experts, quantum_enhanced_routing
    )
    __all__.extend([
        "QuantumRoutingSystem", "QuantumSuperpositionRouter",
        "QuantumAnnealingOptimizer", "get_quantum_router",
        "quantum_route_experts", "quantum_enhanced_routing"
    ])
except ImportError:
    pass

# Distributed optimization system
try:
    from .distributed_optimization import (
        DistributedMoEOptimizer, ClusterNode, NodeType,
        get_distributed_optimizer, distributed_moe_analysis,
        setup_distributed_cluster
    )
    __all__.extend([
        "DistributedMoEOptimizer", "ClusterNode", "NodeType",
        "get_distributed_optimizer", "distributed_moe_analysis",
        "setup_distributed_cluster"
    ])
except ImportError:
    pass

# Advanced caching system
try:
    from .advanced_caching import (
        HierarchicalCacheManager, CacheLevel, CacheState,
        get_cache_manager, cached_moe_analysis
    )
    __all__.extend([
        "HierarchicalCacheManager", "CacheLevel", "CacheState",
        "get_cache_manager", "cached_moe_analysis"
    ])
except ImportError:
    pass

# Progressive Quality Gates - Advanced Features
try:
    from .ai_threat_detection import (
        AIThreatDetectionSystem, ThreatLevel, ThreatCategory,
        get_threat_detection_system, analyze_security_threat,
        start_threat_monitoring, stop_threat_monitoring
    )
    __all__.extend([
        "AIThreatDetectionSystem", "ThreatLevel", "ThreatCategory",
        "get_threat_detection_system", "analyze_security_threat",
        "start_threat_monitoring", "stop_threat_monitoring"
    ])
except ImportError:
    pass

try:
    from .quantum_performance_optimization import (
        QuantumPerformanceOptimizer, OptimizationStrategy, WorkloadType,
        get_performance_optimizer, start_quantum_optimization, stop_quantum_optimization
    )
    __all__.extend([
        "QuantumPerformanceOptimizer", "OptimizationStrategy", "WorkloadType",
        "get_performance_optimizer", "start_quantum_optimization", "stop_quantum_optimization"
    ])
except ImportError:
    pass

try:
    from .chaos_engineering import (
        ChaosEngineeringOrchestrator, ChaosExperimentType, BlastRadiusLevel,
        get_chaos_orchestrator, start_chaos_engineering, stop_chaos_engineering
    )
    __all__.extend([
        "ChaosEngineeringOrchestrator", "ChaosExperimentType", "BlastRadiusLevel", 
        "get_chaos_orchestrator", "start_chaos_engineering", "stop_chaos_engineering"
    ])
except ImportError:
    pass

try:
    from .advanced_observability import (
        AdvancedObservabilitySystem, AlertSeverity, InsightCategory,
        get_observability_system, start_observability_monitoring, stop_observability_monitoring
    )
    __all__.extend([
        "AdvancedObservabilitySystem", "AlertSeverity", "InsightCategory",
        "get_observability_system", "start_observability_monitoring", "stop_observability_monitoring"
    ])
except ImportError:
    pass

try:
    from .enterprise_governance import (
        EnterpriseGovernanceSystem, ComplianceFramework, PolicyType,
        get_governance_system, start_enterprise_governance, stop_enterprise_governance
    )
    __all__.extend([
        "EnterpriseGovernanceSystem", "ComplianceFramework", "PolicyType",
        "get_governance_system", "start_enterprise_governance", "stop_enterprise_governance"
    ])
except ImportError:
    pass

# Progressive Quality Gates Orchestrator
try:
    from .progressive_quality_orchestrator import (
        ProgressiveQualityOrchestrator, QualityGateStatus, QualityGateConfiguration,
        get_progressive_quality_orchestrator, initialize_progressive_quality_gates,
        shutdown_progressive_quality_gates, get_progressive_quality_status,
        run_progressive_quality_assessment
    )
    __all__.extend([
        "ProgressiveQualityOrchestrator", "QualityGateStatus", "QualityGateConfiguration",
        "get_progressive_quality_orchestrator", "initialize_progressive_quality_gates", 
        "shutdown_progressive_quality_gates", "get_progressive_quality_status",
        "run_progressive_quality_assessment"
    ])
except ImportError:
    pass