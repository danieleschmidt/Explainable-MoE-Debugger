"""Main MoE debugger class for real-time model inspection."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Callable
import asyncio
import threading
import time
from contextlib import contextmanager

from .models import (
    DebugSession, HookConfiguration, ModelArchitecture,
    RoutingEvent, ExpertMetrics, PerformanceProfile
)
from .hooks import ModelHooksManager
from .analyzer import MoEAnalyzer
from .profiler import MoEProfiler


class MoEDebugger:
    """Main debugger class for MoE models with real-time visualization."""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = self._create_config(config or {})
        
        # Core components
        self.hooks_manager = ModelHooksManager(model, self.config)
        self.analyzer = MoEAnalyzer(model)
        self.profiler = MoEProfiler(model)
        
        # Session management
        self.current_session: Optional[DebugSession] = None
        self.is_active = False
        
        # Event system
        self.event_callbacks: Dict[str, List[Callable]] = {
            "routing_event": [],
            "performance_update": [],
            "analysis_complete": [],
            "session_start": [],
            "session_end": []
        }
        
        # Model architecture analysis
        self.architecture = self._analyze_architecture()
        
    def _create_config(self, user_config: Dict[str, Any]) -> HookConfiguration:
        """Create hook configuration with defaults."""
        defaults = {
            "enabled_hooks": {
                "router": True,
                "experts": True,
                "attention": False
            },
            "sampling_rate": 0.1,
            "buffer_size": 10000,
            "save_gradients": False,
            "save_activations": True,
            "track_parameters": ["weight", "bias"],
            "memory_limit_mb": 2048
        }
        
        # Merge user config with defaults
        config_dict = {**defaults, **user_config}
        return HookConfiguration(**config_dict)
    
    def _analyze_architecture(self) -> ModelArchitecture:
        """Analyze model architecture to understand MoE structure."""
        num_layers = 0
        num_experts_per_layer = 0
        hidden_size = 0
        
        # Try to extract architecture info from model
        for name, module in self.model.named_modules():
            if "layer" in name.lower():
                layer_num = self._extract_number(name)
                num_layers = max(num_layers, layer_num + 1)
            
            if "expert" in name.lower():
                expert_num = self._extract_number(name)
                num_experts_per_layer = max(num_experts_per_layer, expert_num + 1)
            
            if hasattr(module, "hidden_size"):
                hidden_size = module.hidden_size
            elif hasattr(module, "d_model"):
                hidden_size = module.d_model
        
        # Default values if not found
        if num_layers == 0:
            num_layers = len([m for n, m in self.model.named_modules() if "layer" in n.lower()])
        if num_experts_per_layer == 0:
            num_experts_per_layer = 8  # Common default for MoE models
        if hidden_size == 0:
            hidden_size = 768  # Common default
        
        return ModelArchitecture(
            num_layers=num_layers,
            num_experts_per_layer=num_experts_per_layer,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            vocab_size=getattr(self.model, "vocab_size", 32000),
            max_sequence_length=getattr(self.model, "max_position_embeddings", 2048),
            expert_capacity=2.0
        )
    
    def _extract_number(self, text: str) -> int:
        """Extract number from string."""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[-1]) if numbers else 0
    
    def start_session(self, session_id: Optional[str] = None) -> DebugSession:
        """Start a new debugging session."""
        if self.is_active:
            self.end_session()
        
        session_id = session_id or f"session_{int(time.time())}"
        
        self.current_session = DebugSession(
            session_id=session_id,
            model_name=self.model.__class__.__name__,
            start_time=time.time(),
            end_time=None,
            config=self.config.__dict__
        )
        
        # Register hooks and start monitoring
        self.hooks_manager.register_hooks()
        self.profiler.start_profiling()
        self.is_active = True
        
        # Notify callbacks
        self._emit_event("session_start", self.current_session)
        
        return self.current_session
    
    def end_session(self) -> Optional[DebugSession]:
        """End current debugging session."""
        if not self.is_active or not self.current_session:
            return None
        
        # Stop monitoring and collect final data
        self.hooks_manager.remove_hooks()
        self.profiler.stop_profiling()
        
        # Finalize session data
        self.current_session.end_time = time.time()
        self.current_session.routing_events = self.hooks_manager.get_routing_events()
        self.current_session.performance_profiles = self.profiler.get_profiles()
        
        # Run final analysis
        self.current_session.load_balance_metrics = self.analyzer.analyze_load_balance(
            self.current_session.routing_events
        )
        
        self.is_active = False
        session = self.current_session
        self.current_session = None
        
        # Notify callbacks
        self._emit_event("session_end", session)
        
        return session
    
    @contextmanager
    def trace(self, sequence_id: Optional[str] = None):
        """Context manager for tracing model execution."""
        sequence_id = sequence_id or f"trace_{int(time.time())}"
        
        if not self.is_active:
            self.start_session()
        
        self.hooks_manager.start_sequence(sequence_id)
        
        try:
            yield self
        finally:
            self.hooks_manager.end_sequence()
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for debugging events."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Any):
        """Emit event to registered callbacks."""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Error in callback for {event_type}: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics."""
        if not self.current_session:
            return {}
        
        events = self.hooks_manager.get_routing_events()
        return self.analyzer.compute_routing_statistics(events)
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """Get current expert utilization rates."""
        events = self.hooks_manager.get_routing_events()
        return self.analyzer.compute_expert_utilization(events)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.profiler.get_current_metrics()
    
    def detect_issues(self) -> List[Dict[str, Any]]:
        """Detect common MoE issues."""
        issues = []
        
        if not self.is_active:
            return issues
        
        # Dead expert detection
        dead_experts = self.analyzer.detect_dead_experts(
            self.hooks_manager.get_routing_events()
        )
        if dead_experts:
            issues.append({
                "type": "dead_experts",
                "severity": "warning",
                "message": f"Found {len(dead_experts)} dead experts",
                "experts": dead_experts,
                "suggestion": "Consider increasing router temperature or adjusting expert capacity"
            })
        
        # Router collapse detection
        if self.analyzer.detect_router_collapse(self.hooks_manager.get_routing_events()):
            issues.append({
                "type": "router_collapse",
                "severity": "critical",
                "message": "Router showing signs of collapse (low entropy)",
                "suggestion": "Increase router noise or temperature"
            })
        
        # Load imbalance detection
        load_metrics = self.analyzer.analyze_load_balance(
            self.hooks_manager.get_routing_events()
        )
        if load_metrics and load_metrics.fairness_index < 0.8:
            issues.append({
                "type": "load_imbalance",
                "severity": "warning",
                "message": f"Poor load balancing (fairness: {load_metrics.fairness_index:.2f})",
                "suggestion": "Consider adjusting load balancing loss weight"
            })
        
        return issues
    
    def force_expert_selection(self, expert_ids: List[int]):
        """Force selection of specific experts (for debugging)."""
        # This would require modifying the model's forward pass
        # Implementation depends on specific model architecture
        raise NotImplementedError("Force expert selection requires model-specific implementation")
    
    def set_sampling_rate(self, rate: float):
        """Update sampling rate for data collection."""
        self.config.sampling_rate = max(0.0, min(1.0, rate))
        if hasattr(self.hooks_manager, "config"):
            self.hooks_manager.config.sampling_rate = self.config.sampling_rate
    
    def clear_data(self):
        """Clear all collected debugging data."""
        self.hooks_manager.clear_data()
        self.profiler.clear_data()
        self.analyzer.clear_cache()
    
    def export_session(self, filepath: str):
        """Export current session data to file."""
        if not self.current_session:
            raise ValueError("No active session to export")
        
        import json
        from datetime import datetime
        
        # Convert session to serializable format
        session_data = {
            "session_id": self.current_session.session_id,
            "model_name": self.current_session.model_name,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "config": self.current_session.config,
            "architecture": self.architecture.__dict__,
            "routing_events": [
                {
                    "timestamp": event.timestamp,
                    "layer_idx": event.layer_idx,
                    "token_position": event.token_position,
                    "token": event.token,
                    "expert_weights": event.expert_weights,
                    "selected_experts": event.selected_experts,
                    "routing_confidence": event.routing_confidence
                }
                for event in self.current_session.routing_events
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": self.architecture.__dict__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_class": self.model.__class__.__name__,
            "device": next(self.model.parameters()).device.type,
            "dtype": str(next(self.model.parameters()).dtype),
            "is_active": self.is_active,
            "session_id": self.current_session.session_id if self.current_session else None
        }
    
    def __enter__(self):
        """Support for context manager."""
        self.start_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager."""
        self.end_session()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.is_active:
            self.end_session()