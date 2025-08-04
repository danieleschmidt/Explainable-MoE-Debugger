"""PyTorch hooks for intercepting MoE model execution."""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Any
import threading
import time
from collections import defaultdict, deque

from .models import RoutingEvent, ExpertMetrics, HookConfiguration


class ModelHooksManager:
    """Manages PyTorch hooks for intercepting model execution."""
    
    def __init__(self, model: nn.Module, config: HookConfiguration):
        self.model = model
        self.config = config
        self.hooks = []
        self.routing_events = deque(maxlen=config.buffer_size)
        self.expert_metrics = defaultdict(lambda: defaultdict(float))
        self.sequence_counter = 0
        self.current_sequence_id = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.start_times = {}
        self.layer_timings = defaultdict(list)
        
    def register_hooks(self):
        """Register all configured hooks on the model."""
        self._find_and_hook_modules()
        
    def _find_and_hook_modules(self):
        """Find MoE modules and attach appropriate hooks."""
        for name, module in self.model.named_modules():
            if self._is_router_module(module, name):
                self._hook_router(module, name)
            elif self._is_expert_module(module, name):
                self._hook_expert(module, name)
            elif self._is_attention_module(module, name):
                if self.config.enabled_hooks.get("attention", False):
                    self._hook_attention(module, name)
    
    def _is_router_module(self, module: nn.Module, name: str) -> bool:
        """Identify router/gating modules."""
        router_indicators = ["router", "gate", "gating", "switch"]
        return any(indicator in name.lower() for indicator in router_indicators)
    
    def _is_expert_module(self, module: nn.Module, name: str) -> bool:
        """Identify expert modules."""
        expert_indicators = ["expert", "ffn", "mlp"]
        return any(indicator in name.lower() for indicator in expert_indicators)
    
    def _is_attention_module(self, module: nn.Module, name: str) -> bool:
        """Identify attention modules."""
        attention_indicators = ["attention", "attn", "self_attn"]
        return any(indicator in name.lower() for indicator in attention_indicators)
    
    def _hook_router(self, module: nn.Module, name: str):
        """Add hooks to router/gating modules."""
        layer_idx = self._extract_layer_index(name)
        
        def forward_hook(module, input, output):
            if not self.config.enabled_hooks.get("router", True):
                return
                
            with self.lock:
                # Extract routing weights and decisions
                if isinstance(output, tuple):
                    routing_weights = output[0] if len(output) > 0 else None
                    expert_indices = output[1] if len(output) > 1 else None
                else:
                    routing_weights = output
                    expert_indices = None
                
                if routing_weights is not None:
                    self._process_routing_output(
                        routing_weights, expert_indices, layer_idx, name
                    )
        
        if self.config.save_gradients:
            def backward_hook(module, grad_input, grad_output):
                with self.lock:
                    self._process_routing_gradients(grad_output, layer_idx, name)
        
        # Register hooks
        handle = module.register_forward_hook(forward_hook)
        self.hooks.append(handle)
        
        if self.config.save_gradients:
            handle = module.register_backward_hook(backward_hook)
            self.hooks.append(handle)
    
    def _hook_expert(self, module: nn.Module, name: str):
        """Add hooks to expert modules."""
        layer_idx = self._extract_layer_index(name)
        expert_idx = self._extract_expert_index(name)
        
        def forward_pre_hook(module, input):
            if not self.config.enabled_hooks.get("experts", True):
                return
                
            self.start_times[f"{name}_forward"] = time.perf_counter()
            
            with self.lock:
                # Track expert activation
                self.expert_metrics[layer_idx][expert_idx] += 1
        
        def forward_hook(module, input, output):
            if not self.config.enabled_hooks.get("experts", True):
                return
                
            end_time = time.perf_counter()
            start_time = self.start_times.pop(f"{name}_forward", end_time)
            compute_time = (end_time - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.layer_timings[f"expert_{layer_idx}_{expert_idx}"].append(compute_time)
                
                # Track memory usage if configured
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.expert_metrics[f"memory_{layer_idx}"][expert_idx] = memory_mb
        
        # Register hooks
        handle = module.register_forward_pre_hook(forward_pre_hook)
        self.hooks.append(handle)
        
        handle = module.register_forward_hook(forward_hook)
        self.hooks.append(handle)
    
    def _hook_attention(self, module: nn.Module, name: str):
        """Add hooks to attention modules."""
        layer_idx = self._extract_layer_index(name)
        
        def forward_hook(module, input, output):
            if not self.config.enabled_hooks.get("attention", False):
                return
                
            with self.lock:
                # Process attention weights if available
                if hasattr(module, "attention_weights"):
                    self._process_attention_weights(
                        module.attention_weights, layer_idx, name
                    )
        
        handle = module.register_forward_hook(forward_hook)
        self.hooks.append(handle)
    
    def _process_routing_output(self, routing_weights: torch.Tensor, 
                               expert_indices: Optional[torch.Tensor],
                               layer_idx: int, module_name: str):
        """Process router output to create routing events."""
        if routing_weights.dim() < 2:
            return
            
        batch_size, seq_len = routing_weights.shape[:2]
        
        # Sample based on configured rate
        if torch.rand(1).item() > self.config.sampling_rate:
            return
        
        # Create routing events for each token
        for batch_idx in range(min(batch_size, 1)):  # Process first batch only
            for pos in range(seq_len):
                weights = routing_weights[batch_idx, pos].detach().cpu().numpy()
                
                # Determine selected experts
                if expert_indices is not None:
                    selected = expert_indices[batch_idx, pos].detach().cpu().numpy()
                    if isinstance(selected, (int, float)):
                        selected = [int(selected)]
                    else:
                        selected = selected.tolist()
                else:
                    # Use top-k selection (default k=2)
                    k = min(2, len(weights))
                    _, top_indices = torch.topk(
                        routing_weights[batch_idx, pos], k
                    )
                    selected = top_indices.detach().cpu().numpy().tolist()
                
                # Calculate routing confidence (entropy-based)
                probs = torch.softmax(routing_weights[batch_idx, pos], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                confidence = 1.0 - (entropy / torch.log(torch.tensor(len(weights))))
                
                event = RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=layer_idx,
                    token_position=pos,
                    token=f"token_{pos}",  # Will be replaced with actual tokens
                    expert_weights=weights.tolist(),
                    selected_experts=selected,
                    routing_confidence=confidence,
                    sequence_id=self.current_sequence_id or f"seq_{self.sequence_counter}"
                )
                
                self.routing_events.append(event)
    
    def _process_routing_gradients(self, grad_output: tuple, 
                                  layer_idx: int, module_name: str):
        """Process routing gradients for analysis."""
        if not grad_output or grad_output[0] is None:
            return
            
        grad = grad_output[0]
        grad_norm = torch.norm(grad).item()
        
        # Store gradient information
        self.expert_metrics[f"gradients_{layer_idx}"]["norm"] = grad_norm
    
    def _process_attention_weights(self, attention_weights: torch.Tensor,
                                  layer_idx: int, module_name: str):
        """Process attention weights for correlation analysis."""
        # Store attention patterns for correlation with expert selection
        if attention_weights.dim() >= 3:
            # Average over heads and batch
            avg_attention = attention_weights.mean(dim=(0, 1)).detach().cpu()
            self.expert_metrics[f"attention_{layer_idx}"]["weights"] = avg_attention
    
    def _extract_layer_index(self, module_name: str) -> int:
        """Extract layer index from module name."""
        import re
        match = re.search(r'layer[s]?\.(\d+)', module_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        match = re.search(r'\.(\d+)\.', module_name)
        if match:
            return int(match.group(1))
            
        return 0
    
    def _extract_expert_index(self, module_name: str) -> int:
        """Extract expert index from module name."""
        import re
        match = re.search(r'expert[s]?\.(\d+)', module_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
        match = re.search(r'(\d+)$', module_name)
        if match:
            return int(match.group(1))
            
        return 0
    
    def start_sequence(self, sequence_id: str):
        """Start tracking a new sequence."""
        with self.lock:
            self.current_sequence_id = sequence_id
            self.sequence_counter += 1
    
    def end_sequence(self):
        """End tracking current sequence."""
        with self.lock:
            self.current_sequence_id = None
    
    def get_routing_events(self, limit: Optional[int] = None) -> List[RoutingEvent]:
        """Get collected routing events."""
        with self.lock:
            events = list(self.routing_events)
            if limit:
                events = events[-limit:]
            return events
    
    def get_expert_metrics(self) -> Dict[int, Dict[int, float]]:
        """Get collected expert metrics."""
        with self.lock:
            return dict(self.expert_metrics)
    
    def get_layer_timings(self) -> Dict[str, List[float]]:
        """Get layer timing information."""
        with self.lock:
            return dict(self.layer_timings)
    
    def clear_data(self):
        """Clear all collected data."""
        with self.lock:
            self.routing_events.clear()
            self.expert_metrics.clear()
            self.layer_timings.clear()
            self.start_times.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on destruction."""
        self.remove_hooks()