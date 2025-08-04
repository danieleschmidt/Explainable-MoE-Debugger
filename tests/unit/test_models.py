"""Unit tests for data models."""

import pytest
from datetime import datetime
import json

from moe_debugger.models import (
    RoutingEvent, ExpertMetrics, PerformanceProfile,
    LoadBalanceMetrics, DebugSession, ModelArchitecture,
    VisualizationData, DiagnosticResult, HookConfiguration
)


class TestRoutingEvent:
    """Test RoutingEvent model."""
    
    def test_creation(self):
        """Test basic routing event creation."""
        event = RoutingEvent(
            timestamp=1234567890.0,
            layer_idx=0,
            token_position=5,
            token="hello",
            expert_weights=[0.1, 0.9, 0.0, 0.0],
            selected_experts=[1],
            routing_confidence=0.85
        )
        
        assert event.timestamp == 1234567890.0
        assert event.layer_idx == 0
        assert event.token_position == 5
        assert event.token == "hello"
        assert event.expert_weights == [0.1, 0.9, 0.0, 0.0]
        assert event.selected_experts == [1]
        assert event.routing_confidence == 0.85
        assert event.sequence_id is not None  # Auto-generated
    
    def test_sequence_id_generation(self):
        """Test that sequence IDs are unique."""
        event1 = RoutingEvent(
            timestamp=1.0, layer_idx=0, token_position=0,
            token="test", expert_weights=[1.0], selected_experts=[0],
            routing_confidence=1.0
        )
        
        event2 = RoutingEvent(
            timestamp=2.0, layer_idx=0, token_position=1,
            token="test", expert_weights=[1.0], selected_experts=[0],
            routing_confidence=1.0
        )
        
        assert event1.sequence_id != event2.sequence_id


class TestExpertMetrics:
    """Test ExpertMetrics model."""
    
    def test_creation(self):
        """Test expert metrics creation."""
        metrics = ExpertMetrics(
            expert_id=2,
            layer_idx=1,
            utilization_rate=0.75,
            compute_time_ms=12.5,
            memory_usage_mb=256.0,
            parameter_count=1000000,
            activation_count=150,
            last_activated=datetime(2023, 1, 1, 12, 0, 0)
        )
        
        assert metrics.expert_id == 2
        assert metrics.layer_idx == 1
        assert metrics.utilization_rate == 0.75
        assert metrics.compute_time_ms == 12.5
        assert metrics.memory_usage_mb == 256.0
        assert metrics.parameter_count == 1000000
        assert metrics.activation_count == 150
        assert metrics.last_activated == datetime(2023, 1, 1, 12, 0, 0)
    
    def test_defaults(self):
        """Test default values."""
        metrics = ExpertMetrics(
            expert_id=0,
            layer_idx=0,
            utilization_rate=0.0,
            compute_time_ms=0.0,
            memory_usage_mb=0.0,
            parameter_count=0
        )
        
        assert metrics.activation_count == 0
        assert metrics.last_activated is None


class TestPerformanceProfile:
    """Test PerformanceProfile model."""
    
    def test_creation(self):
        """Test performance profile creation."""
        profile = PerformanceProfile(
            total_inference_time_ms=150.0,
            routing_overhead_ms=15.0,
            expert_compute_times={"expert_0": 50.0, "expert_1": 85.0},
            memory_peak_mb=1024.0,
            cache_hit_rate=0.92,
            token_throughput=500.0
        )
        
        assert profile.total_inference_time_ms == 150.0
        assert profile.routing_overhead_ms == 15.0
        assert profile.expert_compute_times == {"expert_0": 50.0, "expert_1": 85.0}
        assert profile.memory_peak_mb == 1024.0
        assert profile.cache_hit_rate == 0.92
        assert profile.token_throughput == 500.0
        assert isinstance(profile.timestamp, datetime)


class TestLoadBalanceMetrics:
    """Test LoadBalanceMetrics model."""
    
    def test_creation(self):
        """Test load balance metrics creation."""
        metrics = LoadBalanceMetrics(
            expert_loads=[10, 15, 8, 12, 0, 5, 20, 3],
            fairness_index=0.75,
            max_load=20,
            min_load=0,
            coefficient_of_variation=0.85,
            dead_experts=[4],
            overloaded_experts=[6],
            total_tokens_processed=73
        )
        
        assert metrics.expert_loads == [10, 15, 8, 12, 0, 5, 20, 3]
        assert metrics.fairness_index == 0.75
        assert metrics.max_load == 20
        assert metrics.min_load == 0
        assert metrics.coefficient_of_variation == 0.85
        assert metrics.dead_experts == [4]
        assert metrics.overloaded_experts == [6]
        assert metrics.total_tokens_processed == 73


class TestDebugSession:
    """Test DebugSession model."""
    
    def test_creation(self):
        """Test debug session creation."""
        start_time = datetime(2023, 1, 1, 10, 0, 0)
        
        session = DebugSession(
            session_id="test_session_123",
            model_name="TestModel",
            start_time=start_time,
            end_time=None,
            config={"sampling_rate": 0.1}
        )
        
        assert session.session_id == "test_session_123"
        assert session.model_name == "TestModel"
        assert session.start_time == start_time
        assert session.end_time is None
        assert session.config == {"sampling_rate": 0.1}
        assert session.routing_events == []
        assert session.expert_metrics == {}
        assert session.performance_profiles == []
        assert session.load_balance_metrics is None
    
    def test_with_data(self):
        """Test session with routing events and metrics."""
        session = DebugSession(
            session_id="test_session",
            model_name="TestModel",
            start_time=datetime.now(),
            routing_events=[
                RoutingEvent(
                    timestamp=1.0, layer_idx=0, token_position=0,
                    token="test", expert_weights=[1.0], selected_experts=[0],
                    routing_confidence=1.0
                )
            ],
            expert_metrics={
                0: ExpertMetrics(
                    expert_id=0, layer_idx=0, utilization_rate=1.0,
                    compute_time_ms=10.0, memory_usage_mb=100.0,
                    parameter_count=1000
                )
            }
        )
        
        assert len(session.routing_events) == 1
        assert 0 in session.expert_metrics
        assert session.expert_metrics[0].expert_id == 0


class TestModelArchitecture:
    """Test ModelArchitecture model."""
    
    def test_creation(self):
        """Test model architecture creation."""
        arch = ModelArchitecture(
            num_layers=12,
            num_experts_per_layer=8,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=32000,
            max_sequence_length=2048,
            expert_capacity=2.0,
            router_type="top_k",
            expert_types={0: "ffn", 1: "attention"}
        )
        
        assert arch.num_layers == 12
        assert arch.num_experts_per_layer == 8
        assert arch.hidden_size == 768
        assert arch.intermediate_size == 3072
        assert arch.vocab_size == 32000
        assert arch.max_sequence_length == 2048
        assert arch.expert_capacity == 2.0
        assert arch.router_type == "top_k"
        assert arch.expert_types == {0: "ffn", 1: "attention"}
    
    def test_defaults(self):
        """Test default values."""
        arch = ModelArchitecture(
            num_layers=1,
            num_experts_per_layer=2,
            hidden_size=100,
            intermediate_size=200,
            vocab_size=1000,
            max_sequence_length=512,
            expert_capacity=1.0
        )
        
        assert arch.router_type == "top_k"
        assert arch.expert_types == {}


class TestVisualizationData:
    """Test VisualizationData model."""
    
    def test_creation(self):
        """Test visualization data creation."""
        viz_data = VisualizationData(
            routing_matrix=[[0.1, 0.9], [0.8, 0.2]],
            expert_utilization=[0.5, 0.7],
            token_expert_flow=[{"token": "test", "expert": 1}],
            performance_timeline=[{"time": 1.0, "memory": 100}],
            load_balance_chart={"type": "bar", "data": [1, 2, 3]},
            attribution_heatmap=[[0.5, 0.3], [0.7, 0.1]]
        )
        
        assert viz_data.routing_matrix == [[0.1, 0.9], [0.8, 0.2]]
        assert viz_data.expert_utilization == [0.5, 0.7]
        assert viz_data.token_expert_flow == [{"token": "test", "expert": 1}]
        assert viz_data.performance_timeline == [{"time": 1.0, "memory": 100}]
        assert viz_data.load_balance_chart == {"type": "bar", "data": [1, 2, 3]}
        assert viz_data.attribution_heatmap == [[0.5, 0.3], [0.7, 0.1]]
        assert isinstance(viz_data.timestamp, datetime)


class TestDiagnosticResult:
    """Test DiagnosticResult model."""
    
    def test_creation(self):
        """Test diagnostic result creation."""
        result = DiagnosticResult(
            diagnostic_type="dead_experts",
            severity="warning",
            message="Found 2 dead experts",
            affected_experts=[3, 5],
            suggested_actions=["Increase expert capacity", "Adjust routing"],
            metrics={"dead_count": 2, "total_experts": 8}
        )
        
        assert result.diagnostic_type == "dead_experts"
        assert result.severity == "warning"
        assert result.message == "Found 2 dead experts"
        assert result.affected_experts == [3, 5]
        assert result.suggested_actions == ["Increase expert capacity", "Adjust routing"]
        assert result.metrics == {"dead_count": 2, "total_experts": 8}
        assert isinstance(result.timestamp, datetime)


class TestHookConfiguration:
    """Test HookConfiguration model."""
    
    def test_creation(self):
        """Test hook configuration creation."""
        config = HookConfiguration(
            enabled_hooks={"router": True, "experts": False},
            sampling_rate=0.5,
            buffer_size=5000,
            save_gradients=True,
            save_activations=False,
            track_parameters=["weight", "bias"],
            memory_limit_mb=1024
        )
        
        assert config.enabled_hooks == {"router": True, "experts": False}
        assert config.sampling_rate == 0.5
        assert config.buffer_size == 5000
        assert config.save_gradients is True
        assert config.save_activations is False
        assert config.track_parameters == ["weight", "bias"]
        assert config.memory_limit_mb == 1024
    
    def test_defaults(self):
        """Test default values."""
        config = HookConfiguration(
            enabled_hooks={},
            sampling_rate=0.1,
            buffer_size=1000,
            save_gradients=False,
            save_activations=True,
            track_parameters=[]
        )
        
        assert config.memory_limit_mb == 2048  # Default value