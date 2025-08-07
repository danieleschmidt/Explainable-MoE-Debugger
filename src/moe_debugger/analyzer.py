"""Analysis engine for MoE debugging with statistical and behavioral analysis."""

# Try to import numpy, fall back to mock if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def mean(arr): return sum(arr) / len(arr) if arr else 0
        @staticmethod  
        def std(arr): return (sum((x - MockNumpy.mean(arr))**2 for x in arr) / len(arr))**0.5 if arr else 0
        @staticmethod
        def var(arr): return sum((x - MockNumpy.mean(arr))**2 for x in arr) / len(arr) if arr else 0
        @staticmethod
        def sum(arr): return sum(arr)
        @staticmethod
        def max(arr): return max(arr) if arr else 0
        @staticmethod
        def min(arr): return min(arr) if arr else 0
    
    np = MockNumpy()
    NUMPY_AVAILABLE = False
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import math
import warnings

# Try to import torch, fall back to mock if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn
    TORCH_AVAILABLE = False

from .models import (
    RoutingEvent, LoadBalanceMetrics, TokenAttribution, 
    DiagnosticResult, ExpertMetrics
)


class MoEAnalyzer:
    """Core analysis algorithms for MoE debugging and optimization."""
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.cache = {}
        self.analysis_history = []
        
    def analyze_load_balance(self, routing_events: List[RoutingEvent]) -> Optional[LoadBalanceMetrics]:
        """Analyze load balancing across experts."""
        if not routing_events:
            return None
        
        # Count expert selections per layer
        expert_loads = defaultdict(lambda: defaultdict(int))
        total_tokens = 0
        
        for event in routing_events:
            total_tokens += 1
            for expert_id in event.selected_experts:
                expert_loads[event.layer_idx][expert_id] += 1
        
        # Calculate metrics for each layer (using first layer as example)
        if not expert_loads:
            return None
            
        layer_idx = list(expert_loads.keys())[0]
        layer_loads = expert_loads[layer_idx]
        
        # Convert to list format
        max_expert_id = max(layer_loads.keys()) if layer_loads else 0
        load_list = [layer_loads.get(i, 0) for i in range(max_expert_id + 1)]
        
        if not load_list or sum(load_list) == 0:
            return None
        
        # Calculate Jain's fairness index
        load_sum = sum(load_list)
        load_sum_squares = sum(x * x for x in load_list)
        fairness_index = (load_sum ** 2) / (len(load_list) * load_sum_squares) if load_sum_squares > 0 else 0
        
        # Find dead and overloaded experts
        avg_load = load_sum / len(load_list)
        dead_experts = [i for i, load in enumerate(load_list) if load == 0]
        overloaded_experts = [i for i, load in enumerate(load_list) if load > 2 * avg_load]
        
        # Coefficient of variation
        load_mean = np.mean(load_list)
        load_std = np.std(load_list)
        coeff_var = load_std / load_mean if load_mean > 0 else float('inf')
        
        return LoadBalanceMetrics(
            expert_loads=load_list,
            fairness_index=fairness_index,
            max_load=max(load_list),
            min_load=min(load_list),
            coefficient_of_variation=coeff_var,
            dead_experts=dead_experts,
            overloaded_experts=overloaded_experts,
            total_tokens_processed=total_tokens
        )
    
    def detect_dead_experts(self, routing_events: List[RoutingEvent], 
                           threshold: int = 10) -> List[int]:
        """Detect experts that are never or rarely activated."""
        if not routing_events or routing_events is None:
            return []
        
        expert_counts = defaultdict(int)
        
        for event in routing_events:
            for expert_id in event.selected_experts:
                expert_counts[expert_id] += 1
        
        # Find all possible expert IDs
        all_expert_ids = set()
        for event in routing_events:
            all_expert_ids.update(range(len(event.expert_weights)))
        
        # Identify dead experts
        dead_experts = []
        for expert_id in all_expert_ids:
            if expert_counts[expert_id] <= threshold:
                dead_experts.append(expert_id)
        
        return dead_experts
    
    def detect_router_collapse(self, routing_events: List[RoutingEvent],
                              entropy_threshold: float = 0.5) -> bool:
        """Detect when router consistently selects same experts (low entropy)."""
        if not routing_events:
            return False
        
        entropies = []
        
        for event in routing_events:
            # Calculate entropy of routing weights
            weights = np.array(event.expert_weights)
            weights = np.exp(weights) / np.sum(np.exp(weights))  # Softmax
            
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            entropies.append(entropy)
        
        avg_entropy = np.mean(entropies)
        return avg_entropy < entropy_threshold
    
    def compute_routing_statistics(self, routing_events: List[RoutingEvent]) -> Dict[str, Any]:
        """Compute comprehensive routing statistics."""
        if not routing_events:
            return {}
        
        stats = {
            "total_routing_decisions": len(routing_events),
            "unique_sequences": len(set(event.sequence_id for event in routing_events)),
            "layers_analyzed": len(set(event.layer_idx for event in routing_events)),
        }
        
        # Expert selection patterns
        expert_selections = defaultdict(int)
        confidence_scores = []
        
        for event in routing_events:
            for expert_id in event.selected_experts:
                expert_selections[expert_id] += 1
            confidence_scores.append(event.routing_confidence)
        
        # Calculate statistics
        stats.update({
            "avg_confidence": np.mean(confidence_scores),
            "min_confidence": np.min(confidence_scores),
            "max_confidence": np.max(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "most_used_expert": max(expert_selections.items(), key=lambda x: x[1])[0] if expert_selections else None,
            "expert_usage_distribution": dict(expert_selections),
            "avg_experts_per_token": np.mean([len(event.selected_experts) for event in routing_events])
        })
        
        return stats
    
    def compute_expert_utilization(self, routing_events: List[RoutingEvent]) -> Dict[int, float]:
        """Compute utilization rate for each expert."""
        if not routing_events:
            return {}
        
        expert_counts = defaultdict(int)
        total_decisions = len(routing_events)
        
        for event in routing_events:
            for expert_id in event.selected_experts:
                expert_counts[expert_id] += 1
        
        # Calculate utilization rates
        utilization = {}
        for expert_id, count in expert_counts.items():
            utilization[expert_id] = count / total_decisions
        
        return utilization
    
    def analyze_token_attribution(self, routing_events: List[RoutingEvent],
                                token_texts: Optional[List[str]] = None) -> List[TokenAttribution]:
        """Analyze how tokens influence expert selection."""
        attributions = []
        
        for i, event in enumerate(routing_events):
            token_text = token_texts[i] if token_texts and i < len(token_texts) else event.token
            
            # Calculate expert contributions (simplified attribution)
            expert_contributions = {}
            total_weight = sum(event.expert_weights)
            
            for expert_id in event.selected_experts:
                if expert_id < len(event.expert_weights):
                    weight = event.expert_weights[expert_id]
                    expert_contributions[expert_id] = weight / total_weight if total_weight > 0 else 0
            
            attribution = TokenAttribution(
                token=token_text,
                position=event.token_position,
                expert_contributions=expert_contributions,
                attention_weights=[],  # Would need attention weights from model
                gradient_norm=0.0,     # Would need gradient computation
                sequence_id=event.sequence_id
            )
            
            attributions.append(attribution)
        
        return attributions
    
    def compute_expert_similarity(self) -> Dict[Tuple[int, int], float]:
        """Compute pairwise similarity between experts."""
        similarities = {}
        
        if not self.model or not TORCH_AVAILABLE:
            return similarities
        
        # Extract expert parameters
        expert_params = self._extract_expert_parameters()
        
        if not expert_params:
            return similarities
        
        # Compute cosine similarity between expert weight vectors
        expert_ids = list(expert_params.keys())
        
        for i, expert_a in enumerate(expert_ids):
            for j, expert_b in enumerate(expert_ids[i+1:], i+1):
                params_a = expert_params[expert_a]
                params_b = expert_params[expert_b]
                
                # Flatten parameters and compute cosine similarity
                vec_a = torch.cat([p.flatten() for p in params_a])
                vec_b = torch.cat([p.flatten() for p in params_b])
                
                similarity = torch.cosine_similarity(vec_a, vec_b, dim=0).item()
                similarities[(expert_a, expert_b)] = similarity
        
        return similarities
    
    def _extract_expert_parameters(self) -> Dict[int, List]:
        """Extract parameters for each expert."""
        expert_params = defaultdict(list)
        
        if not self.model:
            return dict(expert_params)
        
        for name, param in self.model.named_parameters():
            if "expert" in name.lower():
                # Extract expert ID from parameter name
                expert_id = self._extract_expert_id(name)
                if hasattr(param, 'detach'):
                    expert_params[expert_id].append(param.detach())
                else:
                    expert_params[expert_id].append(param)
        
        return dict(expert_params)
    
    def _extract_expert_id(self, param_name: str) -> int:
        """Extract expert ID from parameter name."""
        import re
        match = re.search(r'expert[s]?\.(\d+)', param_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: look for any number in the name
        match = re.search(r'(\d+)', param_name)
        if match:
            return int(match.group(1))
        
        return 0
    
    def analyze_routing_entropy(self, routing_events: List[RoutingEvent]) -> Dict[str, Any]:
        """Analyze entropy of routing decisions over time."""
        if not routing_events:
            return {}
        
        entropies = []
        timestamps = []
        
        for event in routing_events:
            weights = np.array(event.expert_weights)
            
            # Compute softmax probabilities
            exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
            probs = exp_weights / np.sum(exp_weights)
            
            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
            timestamps.append(event.timestamp)
        
        return {
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "min_entropy": np.min(entropies),
            "max_entropy": np.max(entropies),
            "entropy_trend": np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0,
            "entropy_history": list(zip(timestamps, entropies))
        }
    
    def detect_anomalies(self, routing_events: List[RoutingEvent]) -> List[DiagnosticResult]:
        """Detect anomalies in routing behavior."""
        diagnostics = []
        
        if not routing_events:
            return diagnostics
        
        # Check for routing collapse
        if self.detect_router_collapse(routing_events):
            diagnostics.append(DiagnosticResult(
                diagnostic_type="router_collapse",
                severity="critical",
                message="Router showing signs of collapse (consistently low entropy)",
                affected_experts=[],
                suggested_actions=[
                    "Increase router temperature",
                    "Add noise to router inputs",
                    "Adjust auxiliary loss weights"
                ],
                metrics=self.analyze_routing_entropy(routing_events)
            ))
        
        # Check for dead experts
        dead_experts = self.detect_dead_experts(routing_events)
        if dead_experts:
            diagnostics.append(DiagnosticResult(
                diagnostic_type="dead_experts",
                severity="warning",
                message=f"Found {len(dead_experts)} dead experts",
                affected_experts=dead_experts,
                suggested_actions=[
                    "Increase expert capacity",
                    "Adjust load balancing loss",
                    "Check expert initialization"
                ],
                metrics={"dead_expert_count": len(dead_experts)}
            ))
        
        # Check for extreme load imbalance
        load_metrics = self.analyze_load_balance(routing_events)
        if load_metrics and load_metrics.fairness_index < 0.5:
            diagnostics.append(DiagnosticResult(
                diagnostic_type="load_imbalance",
                severity="warning",
                message=f"Severe load imbalance detected (fairness: {load_metrics.fairness_index:.3f})",
                affected_experts=load_metrics.overloaded_experts,
                suggested_actions=[
                    "Increase load balancing loss weight",
                    "Adjust expert capacity",
                    "Review router architecture"
                ],
                metrics={
                    "fairness_index": load_metrics.fairness_index,
                    "coefficient_of_variation": load_metrics.coefficient_of_variation
                }
            ))
        
        return diagnostics
    
    def generate_optimization_suggestions(self, routing_events: List[RoutingEvent]) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        if not routing_events:
            return ["No routing data available for analysis"]
        
        # Analyze current state
        load_metrics = self.analyze_load_balance(routing_events)
        entropy_stats = self.analyze_routing_entropy(routing_events)
        dead_experts = self.detect_dead_experts(routing_events)
        
        # Generate suggestions based on findings
        if dead_experts:
            suggestions.append(f"Consider reviving {len(dead_experts)} dead experts by adjusting router temperature or expert capacity")
        
        if load_metrics and load_metrics.fairness_index < 0.8:
            suggestions.append("Improve load balancing by increasing the auxiliary loss weight")
        
        if entropy_stats["mean_entropy"] < 1.0:
            suggestions.append("Router entropy is low - consider adding noise or increasing temperature")
        
        if entropy_stats["entropy_trend"] < -0.01:
            suggestions.append("Router entropy is decreasing over time - potential collapse detected")
        
        # Performance suggestions
        avg_experts = np.mean([len(event.selected_experts) for event in routing_events])
        if avg_experts < 1.5:
            suggestions.append("Consider increasing top-k value or expert capacity for better utilization")
        
        if not suggestions:
            suggestions.append("Model routing appears to be functioning normally")
        
        return suggestions
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.cache.clear()
        self.analysis_history.clear()