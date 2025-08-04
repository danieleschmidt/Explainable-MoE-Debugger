"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock
import torch
import torch.nn as nn
from datetime import datetime

from moe_debugger.debugger import MoEDebugger
from moe_debugger.analyzer import MoEAnalyzer
from moe_debugger.profiler import MoEProfiler
from moe_debugger.models import (
    RoutingEvent, ExpertMetrics, PerformanceProfile, 
    HookConfiguration, DebugSession
)
from moe_debugger.database.connection import DatabaseManager
from moe_debugger.cache.manager import CacheManager


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""
    
    def __init__(self, num_layers: int = 2, num_experts: int = 8, hidden_size: int = 768):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Create mock layers
        self.layers = nn.ModuleList([
            MockMoELayer(num_experts, hidden_size) for _ in range(num_layers)
        ])
        
        # Model metadata
        self.config = Mock()
        self.config.num_experts = num_experts
        self.config.num_layers = num_layers
        self.config.hidden_size = hidden_size
    
    def forward(self, input_ids, **kwargs):
        """Mock forward pass."""
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return Mock(last_hidden_state=hidden_states)


class MockMoELayer(nn.Module):
    """Mock MoE layer with router and experts."""
    
    def __init__(self, num_experts: int, hidden_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Mock router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Mock experts
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states):
        """Mock forward pass with routing."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Mock routing weights
        routing_weights = torch.softmax(self.router(hidden_states), dim=-1)
        
        # Mock expert selection (top-2)
        top_k_weights, top_k_indices = torch.topk(routing_weights, 2, dim=-1)
        
        # Mock expert computation
        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_output = self.experts[expert_idx](hidden_states[mask])
                output[mask] += expert_output
        
        return output


@pytest.fixture
def mock_model() -> MockMoEModel:
    """Create a mock MoE model for testing."""
    return MockMoEModel()


@pytest.fixture
def hook_config() -> HookConfiguration:
    """Create test hook configuration."""
    return HookConfiguration(
        enabled_hooks={"router": True, "experts": True, "attention": False},
        sampling_rate=1.0,  # Sample everything in tests
        buffer_size=1000,
        save_gradients=False,
        save_activations=True,
        track_parameters=["weight"],
        memory_limit_mb=1024
    )


@pytest.fixture
def debugger(mock_model: MockMoEModel, hook_config: HookConfiguration) -> MoEDebugger:
    """Create debugger instance for testing."""
    return MoEDebugger(mock_model, hook_config.__dict__)


@pytest.fixture
def analyzer(mock_model: MockMoEModel) -> MoEAnalyzer:
    """Create analyzer instance for testing."""
    return MoEAnalyzer(mock_model)


@pytest.fixture
def profiler(mock_model: MockMoEModel) -> MoEProfiler:
    """Create profiler instance for testing."""
    return MoEProfiler(mock_model)


@pytest.fixture
def sample_routing_events() -> list[RoutingEvent]:
    """Create sample routing events for testing."""
    events = []
    sequence_id = "test_sequence"
    
    for i in range(10):
        event = RoutingEvent(
            timestamp=1234567890.0 + i,
            layer_idx=0,
            token_position=i,
            token=f"token_{i}",
            expert_weights=[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
            selected_experts=[1, 3],  # Top-2 selection
            routing_confidence=0.7 + (i * 0.01),
            sequence_id=sequence_id
        )
        events.append(event)
    
    return events


@pytest.fixture
def sample_expert_metrics() -> list[ExpertMetrics]:
    """Create sample expert metrics for testing."""
    metrics = []
    
    for layer_idx in range(2):
        for expert_id in range(8):
            metric = ExpertMetrics(
                expert_id=expert_id,
                layer_idx=layer_idx,
                utilization_rate=0.1 + (expert_id * 0.1),
                compute_time_ms=10.0 + expert_id,
                memory_usage_mb=100.0 + (expert_id * 10),
                parameter_count=1000000,
                activation_count=expert_id * 10,
                last_activated=datetime.now()
            )
            metrics.append(metric)
    
    return metrics


@pytest.fixture
def sample_performance_profile() -> PerformanceProfile:
    """Create sample performance profile for testing."""
    return PerformanceProfile(
        total_inference_time_ms=100.0,
        routing_overhead_ms=10.0,
        expert_compute_times={"expert_0": 5.0, "expert_1": 7.0},
        memory_peak_mb=512.0,
        cache_hit_rate=0.85,
        token_throughput=1000.0
    )


@pytest.fixture
def temp_db() -> Generator[DatabaseManager, None, None]:
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"
        
    db_manager = DatabaseManager(db_url)
    if db_manager.enabled:
        db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    if db_manager.enabled:
        db_manager.close()
    try:
        import os
        os.unlink(tmp_file.name)
    except (OSError, FileNotFoundError):
        pass


@pytest.fixture
def memory_cache() -> CacheManager:
    """Create memory cache for testing."""
    return CacheManager(cache_type="memory", max_size=1000)


@pytest.fixture
def mock_session() -> DebugSession:
    """Create mock debug session for testing."""
    return DebugSession(
        session_id="test_session_123",
        model_name="MockMoEModel",
        start_time=datetime.now(),
        end_time=None,
        config={"sampling_rate": 1.0}
    )


@pytest.fixture
def sample_tokens() -> list[str]:
    """Sample tokens for testing."""
    return ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing server functionality."""
    websocket = Mock()
    websocket.accept = Mock()
    websocket.send_text = Mock()
    websocket.receive_text = Mock()
    return websocket


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on location."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.name or "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.slow)