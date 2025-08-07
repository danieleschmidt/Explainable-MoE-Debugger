"""Data models for MoE debugging and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class RoutingEvent:
    """Represents a single expert routing decision during model inference."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""
    layer_idx: int = 0
    token_idx: int = 0
    token_text: str = ""
    selected_experts: List[int] = field(default_factory=list)
    routing_weights: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertMetrics:
    """Performance and utilization metrics for individual experts."""
    
    expert_id: int
    layer_idx: int
    total_tokens_processed: int = 0
    average_confidence: float = 0.0
    activation_frequency: float = 0.0
    weight_magnitude: float = 0.0
    gradient_norm: Optional[float] = None
    last_active_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class LoadBalanceMetrics:
    """Load balancing metrics across experts."""
    
    layer_idx: int
    expert_loads: List[int]
    load_variance: float
    coefficient_of_variation: float
    fairness_index: float
    dead_experts: List[int]
    overloaded_experts: List[int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TokenAttribution:
    """Attribution scores showing how tokens influence expert selection."""
    
    token_text: str
    token_idx: int
    expert_attributions: Dict[int, float]
    total_attribution: float
    layer_attributions: Dict[int, float]


@dataclass
class DiagnosticResult:
    """Result of a diagnostic analysis (e.g., dead expert detection)."""
    
    diagnostic_type: str
    severity: str  # "info", "warning", "error"
    title: str
    description: str
    affected_layers: List[int]
    affected_experts: List[int]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PerformanceMetrics:
    """Performance metrics for model inference."""
    
    session_id: str
    timestamp: str
    inference_time_ms: float
    routing_overhead_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    experts_activated: int
    tokens_processed: int
    throughput_tokens_per_sec: float


@dataclass
class SessionInfo:
    """Information about a debugging session."""
    
    session_id: str
    model_name: str
    model_architecture: str
    num_experts: int
    num_layers: int
    created_at: str
    last_active: str
    total_tokens_processed: int
    status: str  # "active", "paused", "completed"


@dataclass
class DebugSession:
    """Complete debugging session with all collected data."""
    
    session_info: SessionInfo
    routing_events: List[RoutingEvent] = field(default_factory=list)
    expert_metrics: List[ExpertMetrics] = field(default_factory=list)
    load_balance_metrics: List[LoadBalanceMetrics] = field(default_factory=list)
    diagnostics: List[DiagnosticResult] = field(default_factory=list)
    performance_metrics: List[PerformanceMetrics] = field(default_factory=list)


@dataclass
class VisualizationData:
    """Structured data prepared for frontend visualization."""
    
    routing_matrix: List[List[float]]  # [seq_len, num_experts]
    expert_utilization: List[float]
    token_expert_flow: List[Dict[str, Any]]
    performance_timeline: List[Dict[str, Any]]
    load_balance_chart: Dict[str, Any]
    attribution_heatmap: List[List[float]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HookConfiguration:
    """Configuration for model hooks and data collection."""
    
    enabled_hooks: Dict[str, bool]
    sampling_rate: float
    buffer_size: int
    save_gradients: bool
    save_activations: bool
    track_parameters: List[str]
    memory_limit_mb: int = 2048