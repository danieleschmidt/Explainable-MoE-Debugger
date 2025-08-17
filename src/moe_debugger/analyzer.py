"""Analysis engine for MoE debugging with statistical and behavioral analysis."""

# Try to import numpy, fall back to mock if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def mean(arr, axis=None): 
            if axis is None:
                return sum(arr) / len(arr) if arr else 0
            elif axis == 0:
                # Column-wise mean for 2D array
                if len(arr) > 0 and hasattr(arr[0], '__len__'):
                    num_cols = len(arr[0])
                    return [sum(arr[i][j] for i in range(len(arr))) / len(arr) for j in range(num_cols)]
                else:
                    return arr
            else:
                return arr
        @staticmethod  
        def std(arr): return (sum((x - MockNumpy.mean(arr))**2 for x in arr) / len(arr))**0.5 if arr else 0
        @staticmethod
        def var(arr): return sum((x - MockNumpy.mean(arr))**2 for x in arr) / len(arr) if arr else 0
        @staticmethod
        def sum(arr, axis=None): 
            if axis is None:
                return sum(arr) if hasattr(arr, '__iter__') else arr
            else:
                return arr
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
            "unique_sequences": len(set(event.session_id for event in routing_events)),
            "layers_analyzed": len(set(event.layer_idx for event in routing_events)),
        }
        
        # Expert selection patterns
        expert_selections = defaultdict(int)
        confidence_scores = []
        
        for event in routing_events:
            for expert_id in event.selected_experts:
                expert_selections[expert_id] += 1
            # Use the first confidence score or 0.0 if empty
            conf_score = event.confidence_scores[0] if event.confidence_scores else 0.0
            confidence_scores.append(conf_score)
        
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
        """Analyze entropy of routing decisions over time with information-theoretic framework."""
        if not routing_events:
            return {}
        
        entropies = []
        timestamps = []
        
        for event in routing_events:
            # Handle both field names for compatibility
            weights = getattr(event, 'expert_weights', None) or getattr(event, 'routing_weights', [])
            
            if NUMPY_AVAILABLE:
                weights = np.array(weights)
                # Compute softmax probabilities
                exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
                probs = exp_weights / np.sum(exp_weights)
                # Compute entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                # Manual computation for mock numpy
                max_weight = max(weights) if weights else 0
                exp_weights = [math.exp(w - max_weight) for w in weights]
                total = sum(exp_weights)
                probs = [w / total for w in exp_weights] if total > 0 else [1.0 / len(weights)] * len(weights)
                entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            
            entropies.append(entropy)
            timestamps.append(event.timestamp)
        
        # Enhanced information-theoretic analysis
        it_metrics = self.compute_information_theoretic_metrics(routing_events)
        
        return {
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "min_entropy": np.min(entropies),
            "max_entropy": np.max(entropies),
            "entropy_trend": np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0,
            "entropy_history": list(zip(timestamps, entropies)),
            **it_metrics
        }
    
    def compute_information_theoretic_metrics(self, routing_events: List[RoutingEvent]) -> Dict[str, Any]:
        """
        Compute comprehensive information-theoretic metrics for expert routing.
        
        Implements Information-Theoretic Expert Analysis (ITEA) framework:
        - Mutual information between inputs and expert selections
        - Information bottleneck principles for optimal routing  
        - Channel capacity analysis for expert communication
        - Novel entropy measures specific to MoE systems
        """
        if not routing_events:
            return {}
        
        # Extract input features and expert selections
        input_features = []
        expert_selections = []
        expert_weights_matrix = []
        
        for event in routing_events:
            if hasattr(event, 'input_features') and event.input_features:
                input_features.append(event.input_features)
            
            # Handle both field names for compatibility
            weights = getattr(event, 'expert_weights', None) or getattr(event, 'routing_weights', [])
            if weights:
                if NUMPY_AVAILABLE:
                    expert_selections.append(np.argmax(weights))
                else:
                    expert_selections.append(weights.index(max(weights)))
                expert_weights_matrix.append(weights)
        
        if NUMPY_AVAILABLE:
            expert_weights_matrix = np.array(expert_weights_matrix)
            num_experts = expert_weights_matrix.shape[1] if expert_weights_matrix.size > 0 else 0
        else:
            num_experts = len(expert_weights_matrix[0]) if expert_weights_matrix and len(expert_weights_matrix) > 0 else 0
        
        # 1. Mutual Information Analysis
        mutual_info_metrics = self._compute_mutual_information(input_features, expert_selections)
        
        # 2. Information Bottleneck Analysis  
        ib_metrics = self._compute_information_bottleneck(expert_weights_matrix, routing_events)
        
        # 3. Channel Capacity Analysis
        channel_metrics = self._compute_channel_capacity(expert_weights_matrix)
        
        # 4. Novel MoE-specific Entropy Measures
        moe_entropy_metrics = self._compute_moe_specific_entropy(expert_weights_matrix, routing_events)
        
        # 5. Information Flow Analysis
        flow_metrics = self._compute_information_flow(expert_weights_matrix, routing_events)
        
        return {
            "mutual_information": mutual_info_metrics,
            "information_bottleneck": ib_metrics,
            "channel_capacity": channel_metrics,
            "moe_entropy_measures": moe_entropy_metrics,
            "information_flow": flow_metrics,
            "num_routing_events": len(routing_events),
            "num_experts": num_experts
        }
    
    def _compute_mutual_information(self, input_features: List, expert_selections: List) -> Dict[str, float]:
        """Compute mutual information I(X;E) between inputs and expert selections."""
        if not input_features or len(input_features) != len(expert_selections):
            return {"mutual_info_input_expert": 0.0, "normalized_mutual_info": 0.0}
        
        # Discretize input features for MI computation
        try:
            input_features_array = np.array(input_features)
            if input_features_array.ndim == 1:
                input_features_array = input_features_array.reshape(-1, 1)
            
            # Simple quantization of input features
            n_bins = min(10, len(set(expert_selections)))
            input_quantized = []
            for i in range(input_features_array.shape[1]):
                feature_col = input_features_array[:, i]
                if np.std(feature_col) > 1e-6:  # Avoid division by zero
                    bins = np.linspace(np.min(feature_col), np.max(feature_col), n_bins + 1)
                    quantized = np.digitize(feature_col, bins) - 1
                    input_quantized.append(quantized)
            
            if input_quantized:
                # Compute MI using histogram-based approach
                input_combined = input_quantized[0]  # Use first feature for simplicity
                mi = self._histogram_mutual_information(input_combined, expert_selections)
                
                # Normalize by max possible MI
                expert_entropy = self._compute_entropy(expert_selections)
                normalized_mi = mi / max(expert_entropy, 1e-10)
                
                return {
                    "mutual_info_input_expert": float(mi),
                    "normalized_mutual_info": float(normalized_mi),
                    "input_entropy": float(self._compute_entropy(input_combined)),
                    "expert_entropy": float(expert_entropy)
                }
        except Exception:
            pass
        
        return {"mutual_info_input_expert": 0.0, "normalized_mutual_info": 0.0}
    
    def _compute_information_bottleneck(self, expert_weights, routing_events: List[RoutingEvent]) -> Dict[str, float]:
        """
        Compute information bottleneck metrics.
        
        Information Bottleneck: R(θ) = I(X;E) - βI(E;Y)
        where X=inputs, E=expert selections, Y=outputs, β=compression parameter
        """
        if len(expert_weights) == 0 if isinstance(expert_weights, list) else (hasattr(expert_weights, 'size') and expert_weights.size == 0):
            return {"ib_objective": 0.0, "compression_ratio": 0.0, "relevance_score": 0.0}
        
        # Compute expert utilization entropy (proxy for compression)
        expert_probs = np.mean(expert_weights, axis=0)
        expert_probs = expert_probs / np.sum(expert_probs)  # Normalize
        compression_entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-10))
        
        # Compute routing consistency (proxy for relevance) 
        routing_consistency = 1.0 - np.mean(np.std(expert_weights, axis=1))
        
        # Information bottleneck objective (simplified)
        beta = 0.1  # Compression parameter
        ib_objective = routing_consistency - beta * compression_entropy
        
        return {
            "ib_objective": float(ib_objective),
            "compression_ratio": float(compression_entropy / np.log(expert_weights.shape[1])),
            "relevance_score": float(routing_consistency),
            "beta_parameter": beta
        }
    
    def _compute_channel_capacity(self, expert_weights) -> Dict[str, float]:
        """Compute channel capacity metrics for expert communication."""
        if len(expert_weights) == 0 if isinstance(expert_weights, list) else (hasattr(expert_weights, 'size') and expert_weights.size == 0):
            return {"channel_capacity": 0.0, "effective_capacity": 0.0, "capacity_utilization": 0.0}
        
        # Theoretical maximum capacity (log of number of experts)
        if NUMPY_AVAILABLE:
            max_capacity = np.log2(expert_weights.shape[1])
        else:
            max_capacity = math.log2(len(expert_weights[0]) if expert_weights and len(expert_weights) > 0 else 1)
        
        # Actual capacity based on weight distribution entropy
        if NUMPY_AVAILABLE:
            mean_weights = np.mean(expert_weights, axis=0)
            mean_weights = mean_weights / np.sum(mean_weights)
            actual_entropy = -np.sum(mean_weights * np.log2(mean_weights + 1e-10))
        else:
            # Manual computation for mock numpy
            num_experts = len(expert_weights[0]) if expert_weights and len(expert_weights) > 0 else 0
            if num_experts > 0:
                mean_weights = [sum(expert_weights[i][j] for i in range(len(expert_weights))) / len(expert_weights) 
                               for j in range(num_experts)]
                total = sum(mean_weights)
                if total > 0:
                    mean_weights = [w / total for w in mean_weights]
                actual_entropy = -sum(w * math.log2(w + 1e-10) for w in mean_weights)
            else:
                mean_weights = []
                actual_entropy = 0.0
        
        # Effective capacity considering routing uncertainty
        if NUMPY_AVAILABLE:
            routing_uncertainty = np.mean([self._compute_entropy(weights) for weights in expert_weights])
        else:
            uncertainty_values = [self._compute_entropy(weights) for weights in expert_weights]
            routing_uncertainty = sum(uncertainty_values) / len(uncertainty_values) if uncertainty_values else 0.0
        effective_capacity = actual_entropy - routing_uncertainty * 0.1
        
        return {
            "channel_capacity": float(max_capacity),
            "effective_capacity": float(max(0, effective_capacity)),
            "capacity_utilization": float(actual_entropy / max_capacity),
            "routing_uncertainty": float(routing_uncertainty)
        }
    
    def _compute_moe_specific_entropy(self, expert_weights, routing_events: List[RoutingEvent]) -> Dict[str, float]:
        """Compute novel entropy measures specific to MoE systems."""
        if len(expert_weights) == 0 if isinstance(expert_weights, list) else (hasattr(expert_weights, 'size') and expert_weights.size == 0):
            return {"load_balance_entropy": 0.0, "specialization_entropy": 0.0, "temporal_entropy": 0.0}
        
        # Load balance entropy - how evenly experts are utilized
        if NUMPY_AVAILABLE:
            expert_utilization = np.mean(expert_weights, axis=0)
            expert_utilization = expert_utilization / np.sum(expert_utilization)
            load_balance_entropy = -np.sum(expert_utilization * np.log(expert_utilization + 1e-10))
        else:
            num_experts = len(expert_weights[0]) if expert_weights and len(expert_weights) > 0 else 0
            if num_experts > 0:
                expert_utilization = [sum(expert_weights[i][j] for i in range(len(expert_weights))) / len(expert_weights) 
                                     for j in range(num_experts)]
                total = sum(expert_utilization)
                if total > 0:
                    expert_utilization = [u / total for u in expert_utilization]
                load_balance_entropy = -sum(u * math.log(u + 1e-10) for u in expert_utilization)
            else:
                load_balance_entropy = 0.0
        
        # Specialization entropy - how specialized each expert is
        specialization_scores = []
        if NUMPY_AVAILABLE:
            for i in range(expert_weights.shape[1]):
                expert_weights_i = expert_weights[:, i]
                if np.std(expert_weights_i) > 1e-6:
                    # Higher specialization = more variance in when expert is used
                    specialization = np.std(expert_weights_i) / (np.mean(expert_weights_i) + 1e-10)
                    specialization_scores.append(specialization)
            specialization_entropy = np.mean(specialization_scores) if specialization_scores else 0.0
        else:
            num_experts = len(expert_weights[0]) if expert_weights and len(expert_weights) > 0 else 0
            for i in range(num_experts):
                expert_weights_i = [expert_weights[j][i] for j in range(len(expert_weights))]
                if len(expert_weights_i) > 1:
                    mean_weight = sum(expert_weights_i) / len(expert_weights_i)
                    std_weight = (sum((w - mean_weight)**2 for w in expert_weights_i) / len(expert_weights_i))**0.5
                    if std_weight > 1e-6:
                        specialization = std_weight / (mean_weight + 1e-10)
                        specialization_scores.append(specialization)
            specialization_entropy = sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0
        
        # Temporal entropy - how routing changes over time
        temporal_entropy = 0.0
        if len(routing_events) > 1:
            temporal_changes = []
            for i in range(1, len(routing_events)):
                if NUMPY_AVAILABLE:
                    prev_selection = np.argmax(expert_weights[i-1])
                    curr_selection = np.argmax(expert_weights[i])
                else:
                    prev_weights = expert_weights[i-1] if isinstance(expert_weights, list) else routing_events[i-1].expert_weights
                    curr_weights = expert_weights[i] if isinstance(expert_weights, list) else routing_events[i].expert_weights
                    prev_selection = prev_weights.index(max(prev_weights))
                    curr_selection = curr_weights.index(max(curr_weights))
                temporal_changes.append(1 if prev_selection != curr_selection else 0)
            
            if temporal_changes:
                if NUMPY_AVAILABLE:
                    change_prob = np.mean(temporal_changes)
                else:
                    change_prob = sum(temporal_changes) / len(temporal_changes)
                if change_prob > 0:
                    if NUMPY_AVAILABLE:
                        temporal_entropy = -(change_prob * np.log(change_prob) + 
                                           (1-change_prob) * np.log(1-change_prob + 1e-10))
                    else:
                        temporal_entropy = -(change_prob * math.log(change_prob) + 
                                           (1-change_prob) * math.log(1-change_prob + 1e-10))
        
        return {
            "load_balance_entropy": float(load_balance_entropy),
            "specialization_entropy": float(specialization_entropy),
            "temporal_entropy": float(temporal_entropy),
            "max_possible_lb_entropy": float(math.log(len(expert_weights[0]) if expert_weights and len(expert_weights) > 0 else 1) if not NUMPY_AVAILABLE else np.log(expert_weights.shape[1]))
        }
    
    def _compute_information_flow(self, expert_weights, routing_events: List[RoutingEvent]) -> Dict[str, float]:
        """Analyze information flow through the expert network."""
        # Check if we have data
        has_data = False
        if isinstance(expert_weights, list):
            has_data = len(expert_weights) > 0 and len(routing_events) >= 2
        else:
            has_data = (hasattr(expert_weights, 'size') and expert_weights.size > 0) and len(routing_events) >= 2
        
        if not has_data:
            return {"flow_rate": 0.0, "flow_efficiency": 0.0, "bottleneck_score": 0.0}
        
        # Information flow rate - how much information flows per routing decision
        flow_rates = []
        num_samples = len(expert_weights) if isinstance(expert_weights, list) else len(routing_events)
        
        for i in range(1, min(num_samples, len(routing_events))):
            # Get distributions from routing events or weights
            if isinstance(expert_weights, list):
                prev_weights = expert_weights[i-1]
                curr_weights = expert_weights[i]
            else:
                prev_weights = routing_events[i-1].expert_weights
                curr_weights = routing_events[i].expert_weights
            
            # Compute KL divergence between consecutive routing distributions
            if NUMPY_AVAILABLE:
                prev_dist = np.array(prev_weights) + 1e-10
                curr_dist = np.array(curr_weights) + 1e-10
                prev_dist = prev_dist / np.sum(prev_dist)
                curr_dist = curr_dist / np.sum(curr_dist)
                kl_div = np.sum(curr_dist * np.log(curr_dist / prev_dist))
            else:
                # Manual computation
                prev_dist = [w + 1e-10 for w in prev_weights]
                curr_dist = [w + 1e-10 for w in curr_weights]
                prev_sum = sum(prev_dist)
                curr_sum = sum(curr_dist)
                prev_dist = [w / prev_sum for w in prev_dist]
                curr_dist = [w / curr_sum for w in curr_dist]
                kl_div = sum(c * math.log(c / p) for c, p in zip(curr_dist, prev_dist))
            
            flow_rates.append(kl_div)
        
        flow_rate = (sum(flow_rates) / len(flow_rates)) if flow_rates else 0.0
        
        # Flow efficiency - how efficiently information is routed
        if isinstance(expert_weights, list):
            active_experts_per_step = [sum(1 for w in weights if w > 0.1) for weights in expert_weights]
            num_experts = len(expert_weights[0]) if expert_weights else 0
        else:
            active_experts_per_step = [sum(1 for w in event.expert_weights if w > 0.1) for event in routing_events]
            num_experts = len(routing_events[0].expert_weights) if routing_events else 0
        
        avg_active_experts = (sum(active_experts_per_step) / len(active_experts_per_step)) if active_experts_per_step else 0.0
        flow_efficiency = avg_active_experts / num_experts if num_experts > 0 else 0.0
        
        # Bottleneck score - identify routing bottlenecks
        if isinstance(expert_weights, list) and expert_weights:
            if NUMPY_AVAILABLE:
                expert_usage_variance = np.var(np.mean(expert_weights, axis=0))
                bottleneck_score = expert_usage_variance / (np.mean(np.mean(expert_weights, axis=0)) + 1e-10)
            else:
                # Manual computation
                num_experts = len(expert_weights[0])
                expert_means = [sum(expert_weights[i][j] for i in range(len(expert_weights))) / len(expert_weights) 
                               for j in range(num_experts)]
                overall_mean = sum(expert_means) / len(expert_means)
                expert_usage_variance = sum((m - overall_mean)**2 for m in expert_means) / len(expert_means)
                bottleneck_score = expert_usage_variance / (overall_mean + 1e-10)
        else:
            bottleneck_score = 0.0
        
        return {
            "flow_rate": float(flow_rate),
            "flow_efficiency": float(flow_efficiency),
            "bottleneck_score": float(bottleneck_score),
            "avg_active_experts": float(avg_active_experts)
        }
    
    def _histogram_mutual_information(self, x, y) -> float:
        """Compute mutual information using histogram-based approach."""
        try:
            if NUMPY_AVAILABLE:
                # Create joint histogram
                x_bins = len(set(x))
                y_bins = len(set(y))
                
                joint_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
                joint_hist = joint_hist + 1e-10  # Smoothing
                joint_prob = joint_hist / np.sum(joint_hist)
                
                # Marginal probabilities
                x_prob = np.sum(joint_prob, axis=1)
                y_prob = np.sum(joint_prob, axis=0)
                
                # Compute mutual information
                mi = 0.0
                for i in range(len(x_prob)):
                    for j in range(len(y_prob)):
                        if joint_prob[i, j] > 1e-10:
                            mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
                
                return max(0.0, mi)
            else:
                # Simple mutual information approximation without numpy
                # Count co-occurrences manually
                unique_x = list(set(x))
                unique_y = list(set(y))
                
                joint_counts = {}
                x_counts = {}
                y_counts = {}
                total = len(x)
                
                for xi, yi in zip(x, y):
                    joint_counts[(xi, yi)] = joint_counts.get((xi, yi), 0) + 1
                    x_counts[xi] = x_counts.get(xi, 0) + 1
                    y_counts[yi] = y_counts.get(yi, 0) + 1
                
                # Compute mutual information
                mi = 0.0
                for xi in unique_x:
                    for yi in unique_y:
                        joint_count = joint_counts.get((xi, yi), 0)
                        if joint_count > 0:
                            p_xy = joint_count / total
                            p_x = x_counts[xi] / total
                            p_y = y_counts[yi] / total
                            mi += p_xy * math.log(p_xy / (p_x * p_y))
                
                return max(0.0, mi)
        except Exception:
            return 0.0
    
    def _compute_entropy(self, data: List) -> float:
        """Compute entropy of discrete data."""
        if not data:
            return 0.0
        
        counts = Counter(data)
        probs = np.array(list(counts.values())) / len(data)
        return -np.sum(probs * np.log(probs + 1e-10))
    
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