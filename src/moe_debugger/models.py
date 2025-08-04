"""Data models for MoE debugging and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class RoutingEvent:
    """Represents a single expert routing decision during model inference."""
    
    timestamp: float
    layer_idx: int
    token_position: int
    token: str
    expert_weights: List[float]
    selected_experts: List[int]
    routing_confidence: float
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExpertMetrics:
    """Performance and utilization metrics for individual experts."""
    
    expert_id: int
    layer_idx: int
    utilization_rate: float
    compute_time_ms: float
    memory_usage_mb: float
    parameter_count: int
    activation_count: int = 0
    last_activated: Optional[datetime] = None


@dataclass
class TokenAttribution:
    """Attribution scores showing how tokens influence expert selection."""
    
    token: str
    position: int
    expert_contributions: Dict[int, float]
    attention_weights: List[float]
    gradient_norm: float
    sequence_id: str


@dataclass
class PerformanceProfile:
    """Complete performance profile for a model inference run."""
    
    total_inference_time_ms: float
    routing_overhead_ms: float
    expert_compute_times: Dict[int, float]
    memory_peak_mb: float
    cache_hit_rate: float
    token_throughput: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LoadBalanceMetrics:
    """Load balancing analysis results for expert utilization."""
    
    expert_loads: List[float]
    fairness_index: float  # Jain's fairness index
    max_load: float
    min_load: float
    coefficient_of_variation: float
    dead_experts: List[int]
    overloaded_experts: List[int]
    total_tokens_processed: int


@dataclass
class DebugSession:
    """Represents a complete debugging session with all collected data."""
    
    session_id: str
    model_name: str
    start_time: float  # Changed from datetime to float for simplicity
    end_time: Optional[float] = None  # Changed from datetime to float 
    routing_events: List[RoutingEvent] = field(default_factory=list)
    expert_metrics: Dict[int, ExpertMetrics] = field(default_factory=dict)
    performance_profiles: List[PerformanceProfile] = field(default_factory=list)
    load_balance_metrics: Optional[LoadBalanceMetrics] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelArchitecture:
    """Describes the architecture of a MoE model for visualization."""
    
    num_layers: int
    num_experts_per_layer: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_sequence_length: int
    expert_capacity: float
    router_type: str = "top_k"
    expert_types: Dict[int, str] = field(default_factory=dict)


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
class DiagnosticResult:
    """Result of a diagnostic analysis (e.g., dead expert detection)."""
    
    diagnostic_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    affected_experts: List[int]
    suggested_actions: List[str]
    metrics: Dict[str, Any]
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