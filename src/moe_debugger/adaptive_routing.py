"""Advanced Adaptive Routing Algorithms for MoE Models.

This module implements novel routing algorithms for dynamic expert selection,
adaptive load balancing, and autonomous expert resurrection mechanisms.

Research Contributions:
1. Entropy-guided Adaptive Routing (EAR) - dynamically adjusts routing based on entropy trends
2. Dead Expert Resurrection Framework (DERF) - automatically revives unused experts
3. Predictive Load Balancing (PLB) - forecasts and prevents load imbalances
4. Multi-objective Routing Optimization (MRO) - balances multiple objectives simultaneously

Authors: Terragon Labs Research Team
License: MIT (with research attribution)
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import asyncio

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(arr): return sum(arr) / len(arr) if arr else 0
        @staticmethod  
        def std(arr): return (sum((x - MockNumpy.mean(arr))**2 for x in arr) / len(arr))**0.5 if arr else 0
        @staticmethod
        def exp(arr): return [math.exp(x) for x in arr]
        @staticmethod
        def sum(arr): return sum(arr)
        @staticmethod
        def array(arr): return list(arr)
        @staticmethod
        def polyfit(x, y, deg): return [0.0] * (deg + 1)
        @staticmethod
        def linalg_norm(arr): return sum(x*x for x in arr)**0.5
    np = MockNumpy()
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn, F
    TORCH_AVAILABLE = False

from .models import RoutingEvent, ExpertMetrics


@dataclass
class AdaptiveRoutingConfig:
    """Configuration for adaptive routing algorithms."""
    # Entropy-guided Adaptive Routing
    entropy_threshold_low: float = 0.5
    entropy_threshold_high: float = 2.5
    adaptation_rate: float = 0.01
    temperature_range: Tuple[float, float] = (0.1, 2.0)
    
    # Dead Expert Resurrection Framework
    resurrection_threshold: int = 100  # tokens without selection
    resurrection_boost: float = 1.5
    resurrection_decay: float = 0.95
    max_resurrection_attempts: int = 5
    
    # Predictive Load Balancing
    prediction_window: int = 50
    imbalance_threshold: float = 0.2
    load_smoothing_factor: float = 0.1
    
    # Multi-objective Routing
    objectives: Dict[str, float] = field(default_factory=lambda: {
        'entropy': 1.0,
        'load_balance': 0.8,
        'performance': 0.6,
        'diversity': 0.4
    })
    
    # Performance monitoring
    enable_real_time_adaptation: bool = True
    adaptation_interval: float = 0.1  # seconds
    history_length: int = 1000


@dataclass
class ExpertState:
    """Tracks the state of individual experts."""
    expert_id: int
    utilization_rate: float = 0.0
    last_selected: int = 0  # token count since last selection
    performance_score: float = 1.0
    resurrection_attempts: int = 0
    adaptive_boost: float = 1.0
    load_trend: List[float] = field(default_factory=list)
    entropy_contribution: float = 0.0


@dataclass
class RoutingDecision:
    """Enhanced routing decision with adaptive features."""
    selected_experts: List[int]
    routing_weights: List[float]
    confidence_score: float
    entropy_score: float
    load_balance_score: float
    adaptation_applied: Dict[str, Any]
    timestamp: float


class EntropyGuidedAdaptiveRouter:
    """Entropy-guided Adaptive Routing (EAR) Algorithm.
    
    Dynamically adjusts routing parameters based on entropy trends
    to prevent router collapse and maintain expert diversity.
    """
    
    def __init__(self, config: AdaptiveRoutingConfig):
        self.config = config
        self.entropy_history = deque(maxlen=config.history_length)
        self.temperature = 1.0
        self.adaptation_history = []
        
    def compute_adaptive_temperature(self, current_entropy: float) -> float:
        """Compute adaptive temperature based on entropy trends."""
        if len(self.entropy_history) < 10:
            return self.temperature
            
        # Analyze entropy trend
        recent_entropies = list(self.entropy_history)[-10:]
        entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
        
        # Adaptive temperature adjustment
        if current_entropy < self.config.entropy_threshold_low:
            # Low entropy - increase temperature to promote diversity
            adaptation = self.config.adaptation_rate * (1 + abs(entropy_trend))
            self.temperature = min(self.config.temperature_range[1], 
                                 self.temperature + adaptation)
        elif current_entropy > self.config.entropy_threshold_high:
            # High entropy - decrease temperature for more focused selection
            adaptation = self.config.adaptation_rate * (1 + abs(entropy_trend))
            self.temperature = max(self.config.temperature_range[0], 
                                 self.temperature - adaptation)
        
        self.entropy_history.append(current_entropy)
        return self.temperature
    
    def apply_adaptive_routing(self, logits: List[float], 
                             expert_states: Dict[int, ExpertState]) -> Tuple[List[float], Dict[str, Any]]:
        """Apply entropy-guided adaptive routing to expert logits."""
        if not logits:
            return logits, {}
            
        # Compute current entropy
        weights = self._softmax(logits, self.temperature)
        current_entropy = self._compute_entropy(weights)
        
        # Update adaptive temperature
        adaptive_temp = self.compute_adaptive_temperature(current_entropy)
        
        # Apply adaptive weights with resurrection boosts
        adapted_logits = []
        for i, logit in enumerate(logits):
            expert_state = expert_states.get(i, ExpertState(i))
            boost = expert_state.adaptive_boost
            adapted_logits.append(logit + math.log(boost))
        
        # Apply temperature scaling
        final_weights = self._softmax(adapted_logits, adaptive_temp)
        
        adaptation_info = {
            'entropy_before': current_entropy,
            'entropy_after': self._compute_entropy(final_weights),
            'temperature': adaptive_temp,
            'boosts_applied': {i: expert_states.get(i, ExpertState(i)).adaptive_boost 
                             for i in range(len(logits))}
        }
        
        return final_weights, adaptation_info
    
    def _softmax(self, logits: List[float], temperature: float = 1.0) -> List[float]:
        """Apply temperature-scaled softmax."""
        scaled_logits = [x / temperature for x in logits]
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(x - max_logit) for x in scaled_logits]
        sum_exp = sum(exp_logits)
        return [x / sum_exp for x in exp_logits]
    
    def _compute_entropy(self, weights: List[float]) -> float:
        """Compute entropy of weight distribution."""
        entropy = 0.0
        for w in weights:
            if w > 1e-10:
                entropy -= w * math.log(w)
        return entropy


class DeadExpertResurrectionFramework:
    """Dead Expert Resurrection Framework (DERF).
    
    Automatically detects and revives underutilized experts through
    adaptive boosting and strategic routing adjustments.
    """
    
    def __init__(self, config: AdaptiveRoutingConfig):
        self.config = config
        self.expert_states: Dict[int, ExpertState] = {}
        self.resurrection_history = []
        self.token_count = 0
        
    def update_expert_usage(self, selected_experts: List[int], 
                           all_expert_ids: List[int]) -> None:
        """Update expert usage statistics."""
        self.token_count += 1
        
        # Update all experts
        for expert_id in all_expert_ids:
            if expert_id not in self.expert_states:
                self.expert_states[expert_id] = ExpertState(expert_id)
            
            state = self.expert_states[expert_id]
            
            if expert_id in selected_experts:
                # Expert was selected - reset last_selected counter
                state.last_selected = 0
                state.utilization_rate = (state.utilization_rate * 0.99 + 0.01)
            else:
                # Expert not selected - increment counter
                state.last_selected += 1
                state.utilization_rate *= 0.999
    
    def identify_dead_experts(self) -> List[int]:
        """Identify experts that need resurrection."""
        dead_experts = []
        
        for expert_id, state in self.expert_states.items():
            if (state.last_selected > self.config.resurrection_threshold and
                state.resurrection_attempts < self.config.max_resurrection_attempts):
                dead_experts.append(expert_id)
        
        return dead_experts
    
    def apply_resurrection_boost(self, expert_id: int) -> float:
        """Apply resurrection boost to dead expert."""
        if expert_id not in self.expert_states:
            return 1.0
            
        state = self.expert_states[expert_id]
        
        if state.last_selected > self.config.resurrection_threshold:
            # Calculate boost based on how long expert has been dead
            deadness_factor = min(state.last_selected / self.config.resurrection_threshold, 5.0)
            boost = self.config.resurrection_boost * deadness_factor
            
            # Apply boost with decay
            state.adaptive_boost = boost
            state.resurrection_attempts += 1
            
            # Log resurrection attempt
            self.resurrection_history.append({
                'expert_id': expert_id,
                'token_count': self.token_count,
                'deadness_factor': deadness_factor,
                'boost_applied': boost
            })
            
            return boost
        
        # Apply decay to existing boosts
        if state.adaptive_boost > 1.0:
            state.adaptive_boost *= self.config.resurrection_decay
            state.adaptive_boost = max(1.0, state.adaptive_boost)
        
        return state.adaptive_boost
    
    def get_resurrection_statistics(self) -> Dict[str, Any]:
        """Get statistics about resurrection attempts."""
        dead_experts = self.identify_dead_experts()
        total_resurrections = len(self.resurrection_history)
        
        success_count = 0
        for expert_id, state in self.expert_states.items():
            if state.resurrection_attempts > 0 and state.last_selected < 10:
                success_count += 1
        
        return {
            'currently_dead': len(dead_experts),
            'dead_expert_ids': dead_experts,
            'total_resurrection_attempts': total_resurrections,
            'successful_resurrections': success_count,
            'resurrection_success_rate': success_count / max(total_resurrections, 1),
            'expert_states': {eid: {
                'utilization_rate': state.utilization_rate,
                'last_selected': state.last_selected,
                'resurrection_attempts': state.resurrection_attempts,
                'adaptive_boost': state.adaptive_boost
            } for eid, state in self.expert_states.items()}
        }


class PredictiveLoadBalancer:
    """Predictive Load Balancing (PLB) Algorithm.
    
    Forecasts expert load patterns and proactively adjusts routing
    to prevent severe load imbalances before they occur.
    """
    
    def __init__(self, config: AdaptiveRoutingConfig):
        self.config = config
        self.load_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.prediction_window)
        )
        self.predictions: Dict[int, float] = {}
        self.balancing_adjustments: Dict[int, float] = defaultdict(float)
        
    def update_load_history(self, expert_loads: Dict[int, float]) -> None:
        """Update load history for all experts."""
        for expert_id, load in expert_loads.items():
            self.load_history[expert_id].append(load)
    
    def predict_future_loads(self) -> Dict[int, float]:
        """Predict future expert loads using trend analysis."""
        predictions = {}
        
        for expert_id, history in self.load_history.items():
            if len(history) < 3:
                predictions[expert_id] = history[-1] if history else 0.0
                continue
            
            # Simple linear trend prediction
            history_list = list(history)
            x = list(range(len(history_list)))
            
            try:
                coeffs = np.polyfit(x, history_list, 1)
                # Predict next value
                next_x = len(history_list)
                predicted_load = coeffs[0] * next_x + coeffs[1]
                predictions[expert_id] = max(0.0, predicted_load)
            except:
                # Fallback to moving average
                predictions[expert_id] = np.mean(history_list[-3:])
        
        self.predictions = predictions
        return predictions
    
    def compute_balancing_adjustments(self, current_loads: Dict[int, float]) -> Dict[int, float]:
        """Compute load balancing adjustments based on predictions."""
        if not current_loads:
            return {}
        
        predictions = self.predict_future_loads()
        
        # Calculate predicted imbalance
        predicted_values = list(predictions.values())
        if not predicted_values:
            return {}
        
        mean_predicted_load = np.mean(predicted_values)
        std_predicted_load = np.std(predicted_values)
        
        adjustments = {}
        
        for expert_id, predicted_load in predictions.items():
            current_load = current_loads.get(expert_id, 0.0)
            
            # Calculate imbalance risk
            if std_predicted_load > 0:
                imbalance_risk = abs(predicted_load - mean_predicted_load) / std_predicted_load
            else:
                imbalance_risk = 0.0
            
            # Apply smoothing adjustment
            if imbalance_risk > self.config.imbalance_threshold:
                if predicted_load > mean_predicted_load:
                    # Expert predicted to be overloaded - apply negative adjustment
                    adjustment = -self.config.load_smoothing_factor * imbalance_risk
                else:
                    # Expert predicted to be underloaded - apply positive adjustment
                    adjustment = self.config.load_smoothing_factor * imbalance_risk
                
                adjustments[expert_id] = adjustment
            else:
                adjustments[expert_id] = 0.0
        
        self.balancing_adjustments = adjustments
        return adjustments
    
    def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get predictive load balancing metrics."""
        if not self.predictions:
            return {}
        
        predicted_values = list(self.predictions.values())
        adjustment_values = list(self.balancing_adjustments.values())
        
        return {
            'predicted_loads': dict(self.predictions),
            'load_balancing_adjustments': dict(self.balancing_adjustments),
            'predicted_mean_load': np.mean(predicted_values),
            'predicted_std_load': np.std(predicted_values),
            'prediction_window_size': self.config.prediction_window,
            'max_adjustment': max(adjustment_values) if adjustment_values else 0.0,
            'min_adjustment': min(adjustment_values) if adjustment_values else 0.0,
            'total_experts_with_adjustments': sum(1 for adj in adjustment_values if abs(adj) > 0.001)
        }


class MultiObjectiveRoutingOptimizer:
    """Multi-objective Routing Optimization (MRO).
    
    Balances multiple objectives simultaneously: entropy, load balance,
    performance, and expert diversity using weighted optimization.
    """
    
    def __init__(self, config: AdaptiveRoutingConfig):
        self.config = config
        self.objective_history: Dict[str, deque] = {
            obj: deque(maxlen=100) for obj in config.objectives.keys()
        }
        self.pareto_front = []
        
    def compute_objective_scores(self, routing_weights: List[float],
                                expert_loads: Dict[int, float],
                                expert_performance: Dict[int, float]) -> Dict[str, float]:
        """Compute scores for all objectives."""
        scores = {}
        
        # Entropy objective (higher is better)
        if routing_weights:
            entropy = -sum(w * math.log(w + 1e-10) for w in routing_weights if w > 1e-10)
            scores['entropy'] = entropy / math.log(len(routing_weights))  # Normalized
        else:
            scores['entropy'] = 0.0
        
        # Load balance objective (higher is better for fairness)
        if expert_loads:
            load_values = list(expert_loads.values())
            if len(load_values) > 1:
                mean_load = np.mean(load_values)
                if mean_load > 0:
                    variance = np.var(load_values)
                    # Jain's fairness index
                    scores['load_balance'] = (sum(load_values) ** 2) / (len(load_values) * sum(x**2 for x in load_values))
                else:
                    scores['load_balance'] = 1.0
            else:
                scores['load_balance'] = 1.0
        else:
            scores['load_balance'] = 1.0
        
        # Performance objective (higher is better)
        if expert_performance:
            selected_experts = [i for i, w in enumerate(routing_weights) if w > 0.01]
            if selected_experts:
                perf_scores = [expert_performance.get(i, 1.0) for i in selected_experts]
                weights_sum = sum(routing_weights[i] for i in selected_experts)
                if weights_sum > 0:
                    weighted_performance = sum(routing_weights[i] * expert_performance.get(i, 1.0) 
                                             for i in selected_experts) / weights_sum
                    scores['performance'] = weighted_performance
                else:
                    scores['performance'] = 1.0
            else:
                scores['performance'] = 1.0
        else:
            scores['performance'] = 1.0
        
        # Diversity objective (higher is better)
        non_zero_weights = sum(1 for w in routing_weights if w > 0.01)
        scores['diversity'] = non_zero_weights / len(routing_weights) if routing_weights else 0.0
        
        # Update history
        for obj, score in scores.items():
            self.objective_history[obj].append(score)
        
        return scores
    
    def compute_multi_objective_score(self, routing_weights: List[float],
                                    expert_loads: Dict[int, float],
                                    expert_performance: Dict[int, float]) -> float:
        """Compute weighted multi-objective score."""
        objective_scores = self.compute_objective_scores(
            routing_weights, expert_loads, expert_performance
        )
        
        # Weighted combination of objectives
        total_score = 0.0
        total_weight = 0.0
        
        for objective, weight in self.config.objectives.items():
            if objective in objective_scores:
                total_score += weight * objective_scores[objective]
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def optimize_routing_decision(self, candidate_weights: List[List[float]],
                                expert_loads: Dict[int, float],
                                expert_performance: Dict[int, float]) -> Tuple[List[float], Dict[str, Any]]:
        """Select optimal routing decision from candidates."""
        if not candidate_weights:
            return [], {}
        
        best_weights = candidate_weights[0]
        best_score = -float('inf')
        best_objectives = {}
        
        for weights in candidate_weights:
            score = self.compute_multi_objective_score(weights, expert_loads, expert_performance)
            
            if score > best_score:
                best_score = score
                best_weights = weights
                best_objectives = self.compute_objective_scores(weights, expert_loads, expert_performance)
        
        optimization_info = {
            'total_score': best_score,
            'objective_scores': best_objectives,
            'candidates_evaluated': len(candidate_weights),
            'objective_weights': dict(self.config.objectives)
        }
        
        return best_weights, optimization_info
    
    def get_objective_trends(self) -> Dict[str, Dict[str, float]]:
        """Get trends for all objectives."""
        trends = {}
        
        for objective, history in self.objective_history.items():
            if len(history) >= 2:
                recent_values = list(history)
                trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                
                trends[objective] = {
                    'current_value': recent_values[-1],
                    'mean_value': np.mean(recent_values),
                    'trend_slope': trend_slope,
                    'trend_direction': 'improving' if trend_slope > 0 else 'degrading' if trend_slope < 0 else 'stable'
                }
            else:
                trends[objective] = {
                    'current_value': list(history)[-1] if history else 0.0,
                    'mean_value': 0.0,
                    'trend_slope': 0.0,
                    'trend_direction': 'unknown'
                }
        
        return trends


class AdaptiveRoutingSystem:
    """Unified Adaptive Routing System.
    
    Integrates all adaptive routing algorithms into a single system
    for comprehensive MoE optimization.
    """
    
    def __init__(self, config: Optional[AdaptiveRoutingConfig] = None):
        self.config = config or AdaptiveRoutingConfig()
        
        # Initialize algorithm components
        self.entropy_router = EntropyGuidedAdaptiveRouter(self.config)
        self.resurrection_framework = DeadExpertResurrectionFramework(self.config)
        self.load_balancer = PredictiveLoadBalancer(self.config)
        self.multi_objective_optimizer = MultiObjectiveRoutingOptimizer(self.config)
        
        # System state
        self.is_active = False
        self.adaptation_thread = None
        self.routing_history = deque(maxlen=self.config.history_length)
        self.performance_metrics = {}
        
        # Real-time adaptation
        if self.config.enable_real_time_adaptation:
            self._start_adaptation_thread()
    
    def process_routing_decision(self, expert_logits: List[float],
                               expert_loads: Dict[int, float],
                               expert_performance: Dict[int, float]) -> RoutingDecision:
        """Process routing decision with all adaptive algorithms."""
        start_time = time.time()
        
        # Step 1: Apply dead expert resurrection boosts
        all_expert_ids = list(range(len(expert_logits)))
        boosted_logits = []
        
        for i, logit in enumerate(expert_logits):
            boost = self.resurrection_framework.apply_resurrection_boost(i)
            boosted_logits.append(logit + math.log(boost))
        
        # Step 2: Apply entropy-guided adaptive routing
        expert_states = self.resurrection_framework.expert_states
        adaptive_weights, adaptation_info = self.entropy_router.apply_adaptive_routing(
            boosted_logits, expert_states
        )
        
        # Step 3: Apply predictive load balancing adjustments
        load_adjustments = self.load_balancer.compute_balancing_adjustments(expert_loads)
        
        adjusted_logits = []
        for i, logit in enumerate(boosted_logits):
            adjustment = load_adjustments.get(i, 0.0)
            adjusted_logits.append(logit + adjustment)
        
        # Recompute weights after load balancing
        final_weights = self.entropy_router._softmax(adjusted_logits, self.entropy_router.temperature)
        
        # Step 4: Multi-objective optimization validation
        candidate_weights = [final_weights]  # In practice, generate multiple candidates
        optimal_weights, optimization_info = self.multi_objective_optimizer.optimize_routing_decision(
            candidate_weights, expert_loads, expert_performance
        )
        
        # Step 5: Select experts based on final weights
        # Use top-k selection (typically k=2 for MoE)
        top_k = 2
        expert_indices = sorted(range(len(optimal_weights)), key=lambda i: optimal_weights[i], reverse=True)
        selected_experts = expert_indices[:top_k]
        
        # Step 6: Update system state
        self.resurrection_framework.update_expert_usage(selected_experts, all_expert_ids)
        self.load_balancer.update_load_history(expert_loads)
        
        # Compute final metrics
        entropy_score = self.entropy_router._compute_entropy(optimal_weights)
        confidence_score = max(optimal_weights) if optimal_weights else 0.0
        load_balance_score = optimization_info.get('objective_scores', {}).get('load_balance', 0.0)
        
        # Create routing decision
        decision = RoutingDecision(
            selected_experts=selected_experts,
            routing_weights=optimal_weights,
            confidence_score=confidence_score,
            entropy_score=entropy_score,
            load_balance_score=load_balance_score,
            adaptation_applied={
                'entropy_adaptation': adaptation_info,
                'resurrection_boosts': {i: self.resurrection_framework.expert_states.get(i, ExpertState(i)).adaptive_boost 
                                      for i in all_expert_ids},
                'load_adjustments': load_adjustments,
                'multi_objective': optimization_info
            },
            timestamp=start_time
        )
        
        self.routing_history.append(decision)
        return decision
    
    def _start_adaptation_thread(self):
        """Start background thread for real-time adaptation."""
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            return
        
        self.is_active = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
    
    def _adaptation_loop(self):
        """Background adaptation loop."""
        while self.is_active:
            try:
                # Perform periodic adaptations
                self._update_performance_metrics()
                self._adaptive_parameter_tuning()
                
                time.sleep(self.config.adaptation_interval)
            except Exception as e:
                print(f"Error in adaptation loop: {e}")
                time.sleep(1.0)
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        if len(self.routing_history) < 10:
            return
        
        recent_decisions = list(self.routing_history)[-10:]
        
        # Compute aggregate metrics
        entropy_scores = [d.entropy_score for d in recent_decisions]
        confidence_scores = [d.confidence_score for d in recent_decisions]
        load_balance_scores = [d.load_balance_score for d in recent_decisions]
        
        self.performance_metrics = {
            'mean_entropy': np.mean(entropy_scores),
            'mean_confidence': np.mean(confidence_scores),
            'mean_load_balance': np.mean(load_balance_scores),
            'routing_stability': np.std(confidence_scores),
            'adaptation_frequency': len([d for d in recent_decisions 
                                       if any(v != 1.0 for v in d.adaptation_applied.get('resurrection_boosts', {}).values())]),
            'timestamp': time.time()
        }
    
    def _adaptive_parameter_tuning(self):
        """Adaptively tune algorithm parameters based on performance."""
        if not self.performance_metrics:
            return
        
        # Tune based on entropy trends
        if self.performance_metrics['mean_entropy'] < 0.5:
            # Low entropy - increase adaptation rate
            self.config.adaptation_rate = min(0.05, self.config.adaptation_rate * 1.1)
        elif self.performance_metrics['mean_entropy'] > 2.0:
            # High entropy - decrease adaptation rate
            self.config.adaptation_rate = max(0.001, self.config.adaptation_rate * 0.9)
        
        # Tune resurrection threshold based on dead expert detection success
        resurrection_stats = self.resurrection_framework.get_resurrection_statistics()
        if resurrection_stats['resurrection_success_rate'] < 0.3:
            # Low success rate - lower threshold for earlier intervention
            self.config.resurrection_threshold = max(50, int(self.config.resurrection_threshold * 0.9))
        elif resurrection_stats['resurrection_success_rate'] > 0.8:
            # High success rate - can afford higher threshold
            self.config.resurrection_threshold = min(200, int(self.config.resurrection_threshold * 1.1))
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'system_performance': self.performance_metrics,
            'entropy_router': {
                'current_temperature': self.entropy_router.temperature,
                'adaptation_history': self.entropy_router.adaptation_history[-10:],
                'entropy_trend': list(self.entropy_router.entropy_history)[-20:]
            },
            'resurrection_framework': self.resurrection_framework.get_resurrection_statistics(),
            'load_balancer': self.load_balancer.get_prediction_metrics(),
            'multi_objective_optimizer': self.multi_objective_optimizer.get_objective_trends(),
            'routing_decisions_processed': len(self.routing_history),
            'config': {
                'adaptation_rate': self.config.adaptation_rate,
                'resurrection_threshold': self.config.resurrection_threshold,
                'temperature_range': self.config.temperature_range,
                'objectives': self.config.objectives
            }
        }
    
    def stop_adaptation(self):
        """Stop real-time adaptation."""
        self.is_active = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_adaptation()


# Research Validation and Benchmarking
class AdaptiveRoutingBenchmark:
    """Benchmarking framework for adaptive routing algorithms."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.adaptive_metrics = {}
        self.test_scenarios = []
    
    def create_test_scenario(self, name: str, expert_count: int, 
                           sequence_length: int, imbalance_factor: float = 1.0) -> Dict[str, Any]:
        """Create a test scenario for benchmarking."""
        return {
            'name': name,
            'expert_count': expert_count,
            'sequence_length': sequence_length,
            'imbalance_factor': imbalance_factor,
            'baseline_results': {},
            'adaptive_results': {}
        }
    
    def run_comparative_study(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comparative study between baseline and adaptive routing."""
        results = {
            'scenarios': scenarios,
            'summary': {},
            'statistical_significance': {}
        }
        
        # Implementation would include:
        # 1. Baseline routing simulation
        # 2. Adaptive routing simulation  
        # 3. Statistical comparison
        # 4. Performance benchmarking
        
        return results


# Export public API
__all__ = [
    'AdaptiveRoutingConfig',
    'ExpertState', 
    'RoutingDecision',
    'EntropyGuidedAdaptiveRouter',
    'DeadExpertResurrectionFramework',
    'PredictiveLoadBalancer',
    'MultiObjectiveRoutingOptimizer',
    'AdaptiveRoutingSystem',
    'AdaptiveRoutingBenchmark'
]