"""Factory for creating MoE debugger instances with improved compatibility."""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn
    TORCH_AVAILABLE = False

from typing import Optional, Dict, Any
from .models import ModelArchitecture

# Conditional import of MoEDebugger
try:
    from .debugger import MoEDebugger
    DEBUGGER_AVAILABLE = True
except ImportError:
    DEBUGGER_AVAILABLE = False
    MoEDebugger = None


class MoEDebuggerFactory:
    """Factory class for creating compatible MoEDebugger instances."""
    
    @staticmethod
    def create_debugger(
        model: Optional[nn.Module] = None, 
        config: Optional[Dict[str, Any]] = None
    ):
        """Create a MoEDebugger instance with compatibility checks."""
        
        # Enhanced config with environment info
        config = config or {}
        config['torch_available'] = TORCH_AVAILABLE
        config['mock_mode'] = not TORCH_AVAILABLE
        
        # If full debugger is not available, use mock
        if not DEBUGGER_AVAILABLE or not TORCH_AVAILABLE:
            return MoEDebuggerFactory._create_mock_debugger(config)
        
        # Create a mock model if none provided
        if model is None:
            model = MoEDebuggerFactory._create_mock_moe_model()
        
        try:
            return MoEDebugger(model, config)
        except Exception as e:
            # Fallback to mock debugger
            return MoEDebuggerFactory._create_mock_debugger(config)
    
    @staticmethod
    def _create_mock_moe_model() -> nn.Module:
        """Create a mock MoE model for testing."""
        
        class MockMoEModel(nn.Module):
            """Mock Mixture of Experts model."""
            
            def __init__(self):
                super().__init__()
                self.vocab_size = 32000
                self.max_position_embeddings = 2048
                self.hidden_size = 768
                self.num_experts = 8
                self.num_layers = 12
                
                # Create mock layers
                for i in range(self.num_layers):
                    layer = nn.Module()
                    # Add mock experts
                    for j in range(self.num_experts):
                        expert = nn.Linear(self.hidden_size, self.hidden_size * 4)
                        setattr(layer, f'expert_{j}', expert)
                    setattr(self, f'layer_{i}', layer)
            
            def forward(self, input_ids, **kwargs):
                # Mock forward pass
                batch_size, seq_len = input_ids.shape if hasattr(input_ids, 'shape') else (1, 10)
                return torch.rand((batch_size, seq_len, self.hidden_size))
        
        return MockMoEModel()
    
    @staticmethod
    def _create_mock_debugger(config: Dict[str, Any]) -> 'MockMoEDebugger':
        """Create a mock debugger for testing when PyTorch is unavailable."""
        return MockMoEDebugger(config)


class MockMoEDebugger:
    """Mock MoE debugger for testing without PyTorch dependencies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_active = False
        self.current_session = None
        self.architecture = ModelArchitecture(
            num_layers=12,
            num_experts_per_layer=8,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=32000,
            max_sequence_length=2048,
            expert_capacity=2.0
        )
    
    def start_session(self, session_id: Optional[str] = None):
        """Start a mock debugging session."""
        from .models import DebugSession, SessionInfo
        import time
        
        session_id = session_id or f"mock_session_{int(time.time())}"
        
        session_info = SessionInfo(
            session_id=session_id,
            model_name="MockMoEModel", 
            model_architecture="MockMoE",
            num_experts=8,
            num_layers=12,
            created_at=str(time.time()),
            last_active=str(time.time()),
            total_tokens_processed=0,
            status="active"
        )
        
        self.current_session = DebugSession(session_info=session_info)
        self.is_active = True
        return self.current_session
    
    def end_session(self):
        """End mock debugging session."""
        if self.current_session:
            import time
            self.current_session.session_info.status = "completed"
            self.current_session.session_info.last_active = str(time.time())
            self.is_active = False
        return self.current_session
    
    def get_routing_stats(self):
        """Get mock routing statistics."""
        return {
            'total_events': 100,
            'dead_experts': [2, 5],
            'overloaded_experts': [0, 3, 7],
            'average_confidence': 0.75,
            'load_balance_coefficient': 0.85
        }
    
    def get_expert_metrics(self):
        """Get mock expert metrics."""
        return {
            'utilization_distribution': [0.1, 0.15, 0.0, 0.2, 0.1, 0.0, 0.25, 0.2],
            'average_activations': 2.5,
            'total_tokens_processed': 1000
        }
    
    def get_performance_metrics(self):
        """Get mock performance metrics."""
        return {
            'average_inference_time': 0.05,
            'memory_usage_mb': 1024,
            'throughput_tokens_per_sec': 2000,
            'cache_hit_rate': 0.8
        }
    
    def profile(self):
        """Mock profiling context manager."""
        return MockProfileContext()


class MockProfileContext:
    """Mock profiling context manager."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass