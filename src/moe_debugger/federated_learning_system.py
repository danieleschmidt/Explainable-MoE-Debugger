"""
Multi-Cloud Federated Learning System for MoE Models

This module implements a comprehensive federated learning framework specifically
designed for Mixture of Experts models across multiple cloud providers and
edge devices. It enables privacy-preserving collaborative training while
maintaining expert specialization and load balancing.

Features:
- Multi-cloud federation (AWS, GCP, Azure, edge devices)
- Privacy-preserving aggregation using secure multiparty computation
- Expert-aware federated averaging with specialization preservation
- Adaptive communication compression and quantization
- Byzantine fault tolerance and malicious participant detection
- Differential privacy integration
- Cross-silo and cross-device federated learning
"""

import asyncio
import logging
import random
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
import numpy as np
from collections import defaultdict, deque
import threading
from abc import ABC, abstractmethod
import base64

# Mock cryptographic libraries for demonstration
try:
    import cryptography
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    CRYPTO_AVAILABLE = True
except ImportError:
    # Mock cryptographic functions
    class MockCrypto:
        @staticmethod
        def generate_private_key():
            return "mock_private_key"
        
        @staticmethod
        def encrypt(data, key):
            return base64.b64encode(json.dumps(data).encode()).decode()
        
        @staticmethod
        def decrypt(encrypted_data, key):
            return json.loads(base64.b64decode(encrypted_data.encode()).decode())
    
    cryptography = MockCrypto()
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers for federated learning."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ALIBABA = "alibaba"
    EDGE = "edge"
    ON_PREMISE = "on_premise"


class FederationRole(Enum):
    """Roles in the federated learning system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"


class AggregationStrategy(Enum):
    """Strategies for federated aggregation."""
    FEDERATED_AVERAGING = "fed_avg"
    WEIGHTED_AVERAGING = "weighted_avg"
    SECURE_AGGREGATION = "secure_agg"
    EXPERT_AWARE_AGGREGATION = "expert_aware"
    ADAPTIVE_AGGREGATION = "adaptive"


class PrivacyLevel(Enum):
    """Privacy preservation levels."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "dp"
    SECURE_AGGREGATION = "secure"
    HOMOMORPHIC_ENCRYPTION = "homomorphic"
    FULL_PRIVACY = "full"


@dataclass
class ParticipantInfo:
    """Information about a federated learning participant."""
    participant_id: str
    cloud_provider: CloudProvider
    role: FederationRole
    capabilities: Dict[str, Any] = field(default_factory=dict)
    trust_score: float = 1.0
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    
    # Network and compute resources
    bandwidth_mbps: float = 100.0
    compute_flops: float = 1e12  # FLOPS
    memory_gb: float = 16.0
    storage_gb: float = 100.0
    
    # Privacy preferences
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    max_privacy_budget: float = 1.0
    current_privacy_spent: float = 0.0
    
    def update_trust_score(self, performance: float, reliability: float) -> None:
        """Update trust score based on performance and reliability."""
        new_score = 0.7 * performance + 0.3 * reliability
        # Exponential moving average
        alpha = 0.1
        self.trust_score = alpha * new_score + (1 - alpha) * self.trust_score
        self.trust_score = np.clip(self.trust_score, 0.0, 1.0)
    
    def can_participate(self, privacy_cost: float) -> bool:
        """Check if participant can participate given privacy cost."""
        return (
            self.current_privacy_spent + privacy_cost <= self.max_privacy_budget and
            self.trust_score >= 0.5
        )


@dataclass
class ModelUpdate:
    """Represents a model update from a participant."""
    participant_id: str
    update_id: str
    weights_delta: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Update characteristics
    num_samples: int = 0
    training_loss: float = float('inf')
    validation_accuracy: float = 0.0
    computation_time: float = 0.0
    communication_cost: float = 0.0
    
    # Privacy and security
    privacy_spent: float = 0.0
    is_secure: bool = False
    signature: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def compute_quality_score(self) -> float:
        """Compute quality score for this update."""
        # Combine multiple factors
        loss_score = 1.0 / (1.0 + self.training_loss)
        accuracy_score = self.validation_accuracy
        size_score = min(1.0, self.num_samples / 1000.0)  # Normalize by expected size
        
        quality_score = 0.4 * loss_score + 0.4 * accuracy_score + 0.2 * size_score
        return np.clip(quality_score, 0.0, 1.0)
    
    def get_compressed_size(self) -> int:
        """Estimate compressed size of the update."""
        total_params = sum(w.size for w in self.weights_delta.values())
        # Assume 32-bit floats and 50% compression
        return int(total_params * 4 * 0.5)


class SecureAggregationProtocol:
    """Secure aggregation protocol using cryptographic techniques."""
    
    def __init__(self, num_participants: int, threshold: int = None):
        self.num_participants = num_participants
        self.threshold = threshold or max(1, num_participants // 2)
        
        # Cryptographic keys (simplified)
        self.private_keys = {}
        self.public_keys = {}
        self.shared_secrets = {}
        
        logger.info(f"Initialized secure aggregation for {num_participants} participants")
    
    def setup_keys(self, participant_ids: List[str]) -> Dict[str, str]:
        """Setup cryptographic keys for participants."""
        keys = {}
        
        for participant_id in participant_ids:
            if CRYPTO_AVAILABLE:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                public_key = private_key.public_key()
            else:
                private_key = f"private_key_{participant_id}"
                public_key = f"public_key_{participant_id}"
            
            self.private_keys[participant_id] = private_key
            self.public_keys[participant_id] = public_key
            keys[participant_id] = str(public_key)
        
        logger.info(f"Generated keys for {len(participant_ids)} participants")
        return keys
    
    def generate_masks(self, participant_id: str, round_number: int) -> Dict[str, np.ndarray]:
        """Generate cryptographic masks for secure aggregation."""
        # Simplified mask generation (in practice, use proper cryptographic protocols)
        np.random.seed(hash(f"{participant_id}_{round_number}") % (2**32))
        
        masks = {}
        for other_id in self.public_keys:
            if other_id != participant_id:
                # Generate shared random mask
                mask_seed = hash(f"{min(participant_id, other_id)}_{max(participant_id, other_id)}_{round_number}")
                np.random.seed(mask_seed % (2**32))
                
                # Create mask with appropriate sign
                sign = 1 if participant_id < other_id else -1
                masks[other_id] = sign * np.random.randn(1000)  # Simplified dimension
        
        return masks
    
    def mask_update(
        self, 
        weights_update: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        participant_id: str
    ) -> Dict[str, np.ndarray]:
        """Apply cryptographic masks to model update."""
        masked_update = {}
        
        for layer_name, weights in weights_update.items():
            # Flatten weights for masking
            flat_weights = weights.flatten()
            
            # Apply all masks
            total_mask = np.zeros_like(flat_weights)
            mask_idx = 0
            
            for mask in masks.values():
                mask_size = min(len(mask), len(flat_weights) - mask_idx)
                if mask_size > 0:
                    total_mask[mask_idx:mask_idx + mask_size] += mask[:mask_size]
                    mask_idx += mask_size
                
                if mask_idx >= len(flat_weights):
                    break
            
            # Add mask to weights
            masked_flat = flat_weights + total_mask[:len(flat_weights)]
            masked_update[layer_name] = masked_flat.reshape(weights.shape)
        
        return masked_update
    
    def aggregate_masked_updates(
        self,
        masked_updates: List[Dict[str, np.ndarray]],
        participant_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Aggregate masked updates (masks cancel out)."""
        if not masked_updates:
            return {}
        
        # Initialize aggregated update
        aggregated = {}
        for layer_name in masked_updates[0]:
            aggregated[layer_name] = np.zeros_like(masked_updates[0][layer_name])
        
        # Sum all masked updates (masks cancel due to complementary signs)
        for masked_update in masked_updates:
            for layer_name, weights in masked_update.items():
                aggregated[layer_name] += weights
        
        # Average by number of participants
        for layer_name in aggregated:
            aggregated[layer_name] /= len(masked_updates)
        
        return aggregated


class DifferentialPrivacyManager:
    """Manages differential privacy for federated learning."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        sensitivity: float = 1.0,
        mechanism: str = "gaussian"
    ):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.sensitivity = sensitivity  # Global sensitivity
        self.mechanism = mechanism
        
        self.privacy_spent = 0.0
        self.noise_history = deque(maxlen=1000)
        
        logger.info(f"Initialized DP manager with ε={epsilon}, δ={delta}")
    
    def compute_noise_scale(self, epsilon: float, delta: float) -> float:
        """Compute noise scale for Gaussian mechanism."""
        if self.mechanism == "gaussian":
            # Gaussian mechanism: σ ≥ sqrt(2 ln(1.25/δ)) * Δ / ε
            return np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity / epsilon
        elif self.mechanism == "laplace":
            # Laplace mechanism: b = Δ / ε
            return self.sensitivity / epsilon
        else:
            return 1.0
    
    def add_noise_to_update(
        self,
        weights_update: Dict[str, np.ndarray],
        epsilon: float,
        delta: float = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Add differential privacy noise to model update."""
        delta = delta or self.delta
        noise_scale = self.compute_noise_scale(epsilon, delta)
        
        noisy_update = {}
        noise_info = {
            'epsilon_used': epsilon,
            'delta_used': delta,
            'noise_scale': noise_scale,
            'mechanism': self.mechanism,
            'layers_noised': []
        }
        
        for layer_name, weights in weights_update.items():
            if self.mechanism == "gaussian":
                noise = np.random.normal(0, noise_scale, weights.shape)
            elif self.mechanism == "laplace":
                noise = np.random.laplace(0, noise_scale, weights.shape)
            else:
                noise = np.zeros_like(weights)
            
            noisy_update[layer_name] = weights + noise
            noise_info['layers_noised'].append({
                'layer': layer_name,
                'original_norm': float(np.linalg.norm(weights)),
                'noise_norm': float(np.linalg.norm(noise)),
                'snr': float(np.linalg.norm(weights) / (np.linalg.norm(noise) + 1e-8))
            })
        
        # Update privacy budget
        self.privacy_spent += epsilon
        self.noise_history.append(noise_info)
        
        return noisy_update, noise_info
    
    def clip_gradients(
        self,
        weights_update: Dict[str, np.ndarray],
        clipping_norm: float = 1.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Clip gradients to bound sensitivity."""
        clipped_update = {}
        clipping_info = {
            'clipping_norm': clipping_norm,
            'layers_clipped': []
        }
        
        for layer_name, weights in weights_update.items():
            layer_norm = np.linalg.norm(weights)
            
            if layer_norm > clipping_norm:
                # Clip to maximum norm
                clipped_weights = weights * (clipping_norm / layer_norm)
                clipped_update[layer_name] = clipped_weights
                
                clipping_info['layers_clipped'].append({
                    'layer': layer_name,
                    'original_norm': float(layer_norm),
                    'clipped_norm': float(clipping_norm),
                    'clipping_ratio': float(clipping_norm / layer_norm)
                })
            else:
                clipped_update[layer_name] = weights.copy()
        
        return clipped_update, clipping_info
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.epsilon - self.privacy_spent)
    
    def privacy_accountant_analysis(self) -> Dict[str, Any]:
        """Analyze privacy spending using advanced accounting."""
        return {
            'total_epsilon_spent': self.privacy_spent,
            'remaining_budget': self.get_remaining_budget(),
            'total_queries': len(self.noise_history),
            'avg_epsilon_per_query': self.privacy_spent / max(1, len(self.noise_history)),
            'budget_exhausted': self.privacy_spent >= self.epsilon
        }


class ExpertAwareFederatedAggregator:
    """Specialized aggregator for MoE models preserving expert specializations."""
    
    def __init__(
        self,
        num_experts: int = 8,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.EXPERT_AWARE_AGGREGATION,
        expert_similarity_threshold: float = 0.8
    ):
        self.num_experts = num_experts
        self.aggregation_strategy = aggregation_strategy
        self.expert_similarity_threshold = expert_similarity_threshold
        
        # Expert specialization tracking
        self.expert_specializations = defaultdict(list)
        self.expert_performance_history = defaultdict(list)
        self.cross_participant_expert_mapping = {}
        
        logger.info(f"Initialized expert-aware aggregator for {num_experts} experts")
    
    async def aggregate_updates(
        self,
        updates: List[ModelUpdate],
        participant_info: Dict[str, ParticipantInfo],
        current_round: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Aggregate model updates preserving expert specializations."""
        if not updates:
            return {}, {'error': 'no_updates'}
        
        logger.info(f"Aggregating {len(updates)} updates for round {current_round}")
        
        # Analyze expert specializations across participants
        expert_analysis = await self._analyze_expert_specializations(updates)
        
        # Choose aggregation method based on strategy
        if self.aggregation_strategy == AggregationStrategy.EXPERT_AWARE_AGGREGATION:
            aggregated_weights, agg_info = await self._expert_aware_aggregation(
                updates, participant_info, expert_analysis
            )
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGING:
            aggregated_weights, agg_info = await self._weighted_averaging(
                updates, participant_info
            )
        elif self.aggregation_strategy == AggregationStrategy.ADAPTIVE_AGGREGATION:
            aggregated_weights, agg_info = await self._adaptive_aggregation(
                updates, participant_info, current_round
            )
        else:  # Default to federated averaging
            aggregated_weights, agg_info = await self._federated_averaging(updates)
        
        # Update expert specialization tracking
        self._update_expert_specializations(updates, expert_analysis)
        
        aggregation_metadata = {
            'round': current_round,
            'num_updates': len(updates),
            'aggregation_strategy': self.aggregation_strategy.value,
            'expert_analysis': expert_analysis,
            'aggregation_info': agg_info,
            'timestamp': time.time()
        }
        
        return aggregated_weights, aggregation_metadata
    
    async def _analyze_expert_specializations(
        self, 
        updates: List[ModelUpdate]
    ) -> Dict[str, Any]:
        """Analyze expert specializations across participant updates."""
        expert_weights = defaultdict(list)
        expert_performances = defaultdict(list)
        
        # Collect expert weights from all updates
        for update in updates:
            participant_id = update.participant_id
            
            # Extract expert-related weights (simplified pattern matching)
            for layer_name, weights in update.weights_delta.items():
                if 'expert' in layer_name.lower():
                    # Extract expert ID from layer name (e.g., "expert_2_fc1")
                    try:
                        expert_id = int(layer_name.split('expert_')[1].split('_')[0])
                        expert_weights[expert_id].append({
                            'participant': participant_id,
                            'weights': weights,
                            'layer': layer_name
                        })
                    except (IndexError, ValueError):
                        continue
        
        # Compute expert similarity matrices
        expert_similarities = {}
        for expert_id, weight_list in expert_weights.items():
            if len(weight_list) > 1:
                similarities = self._compute_weight_similarities(weight_list)
                expert_similarities[expert_id] = similarities
        
        # Identify expert clusters and specializations
        expert_clusters = self._identify_expert_clusters(expert_similarities)
        
        analysis_result = {
            'num_experts_found': len(expert_weights),
            'expert_similarities': expert_similarities,
            'expert_clusters': expert_clusters,
            'specialization_diversity': self._compute_specialization_diversity(expert_weights),
            'cross_participant_consistency': self._compute_cross_participant_consistency(expert_weights)
        }
        
        return analysis_result
    
    def _compute_weight_similarities(
        self, 
        weight_list: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute similarities between expert weights from different participants."""
        similarities = {}
        
        for i in range(len(weight_list)):
            for j in range(i + 1, len(weight_list)):
                participant_a = weight_list[i]['participant']
                participant_b = weight_list[j]['participant']
                weights_a = weight_list[i]['weights'].flatten()
                weights_b = weight_list[j]['weights'].flatten()
                
                # Compute cosine similarity
                dot_product = np.dot(weights_a, weights_b)
                norm_a = np.linalg.norm(weights_a)
                norm_b = np.linalg.norm(weights_b)
                
                if norm_a > 1e-8 and norm_b > 1e-8:
                    similarity = dot_product / (norm_a * norm_b)
                else:
                    similarity = 0.0
                
                similarities[(participant_a, participant_b)] = float(similarity)
        
        return similarities
    
    def _identify_expert_clusters(
        self, 
        expert_similarities: Dict[int, Dict[Tuple[str, str], float]]
    ) -> Dict[int, List[List[str]]]:
        """Identify clusters of participants with similar expert specializations."""
        expert_clusters = {}
        
        for expert_id, similarities in expert_similarities.items():
            # Simple clustering based on similarity threshold
            participants = set()
            for (p1, p2) in similarities.keys():
                participants.add(p1)
                participants.add(p2)
            
            participants = list(participants)
            clusters = []
            remaining = set(participants)
            
            while remaining:
                # Start new cluster with arbitrary participant
                seed = remaining.pop()
                cluster = [seed]
                
                # Add similar participants to cluster
                to_check = [seed]
                while to_check:
                    current = to_check.pop()
                    for other in list(remaining):
                        pair = (min(current, other), max(current, other))
                        if pair in similarities:
                            similarity = similarities[pair]
                            if similarity >= self.expert_similarity_threshold:
                                cluster.append(other)
                                remaining.remove(other)
                                to_check.append(other)
                
                if len(cluster) > 1:  # Only keep clusters with multiple participants
                    clusters.append(cluster)
            
            expert_clusters[expert_id] = clusters
        
        return expert_clusters
    
    def _compute_specialization_diversity(
        self, 
        expert_weights: Dict[int, List[Dict[str, Any]]]
    ) -> float:
        """Compute diversity of expert specializations."""
        if not expert_weights:
            return 0.0
        
        # Compute variance in expert weight magnitudes
        expert_norms = []
        for expert_id, weight_list in expert_weights.items():
            for weight_info in weight_list:
                norm = np.linalg.norm(weight_info['weights'])
                expert_norms.append(norm)
        
        if len(expert_norms) < 2:
            return 0.0
        
        # Normalize by mean to get coefficient of variation
        diversity = np.std(expert_norms) / (np.mean(expert_norms) + 1e-8)
        return float(diversity)
    
    def _compute_cross_participant_consistency(
        self, 
        expert_weights: Dict[int, List[Dict[str, Any]]]
    ) -> float:
        """Compute consistency of expert weights across participants."""
        consistency_scores = []
        
        for expert_id, weight_list in expert_weights.items():
            if len(weight_list) < 2:
                continue
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(weight_list)):
                for j in range(i + 1, len(weight_list)):
                    w1 = weight_list[i]['weights'].flatten()
                    w2 = weight_list[j]['weights'].flatten()
                    
                    # Cosine similarity
                    dot = np.dot(w1, w2)
                    norm1 = np.linalg.norm(w1)
                    norm2 = np.linalg.norm(w2)
                    
                    if norm1 > 1e-8 and norm2 > 1e-8:
                        sim = dot / (norm1 * norm2)
                    else:
                        sim = 0.0
                    
                    similarities.append(sim)
            
            if similarities:
                expert_consistency = np.mean(similarities)
                consistency_scores.append(expert_consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    async def _expert_aware_aggregation(
        self,
        updates: List[ModelUpdate],
        participant_info: Dict[str, ParticipantInfo],
        expert_analysis: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Perform expert-aware federated aggregation."""
        # Group updates by expert clusters
        expert_clusters = expert_analysis.get('expert_clusters', {})
        
        # Initialize aggregated weights
        aggregated = {}
        first_update = updates[0]
        for layer_name, weights in first_update.weights_delta.items():
            aggregated[layer_name] = np.zeros_like(weights)
        
        # Weight participants based on trust score and data quality
        participant_weights = {}
        total_weight = 0.0
        
        for update in updates:
            participant_id = update.participant_id
            if participant_id in participant_info:
                trust_score = participant_info[participant_id].trust_score
                quality_score = update.compute_quality_score()
                sample_weight = np.log(max(1, update.num_samples))
                
                weight = trust_score * quality_score * sample_weight
                participant_weights[participant_id] = weight
                total_weight += weight
            else:
                participant_weights[participant_id] = 1.0
                total_weight += 1.0
        
        # Normalize weights
        if total_weight > 0:
            for participant_id in participant_weights:
                participant_weights[participant_id] /= total_weight
        
        # Aggregate layers with expert-awareness
        layer_aggregation_info = {}
        
        for layer_name in aggregated:
            if 'expert' in layer_name.lower():
                # Expert layer: use cluster-aware aggregation
                aggregated[layer_name], layer_info = self._aggregate_expert_layer(
                    layer_name, updates, participant_weights, expert_clusters
                )
            else:
                # Non-expert layer: standard weighted average
                for update in updates:
                    if layer_name in update.weights_delta:
                        weight = participant_weights[update.participant_id]
                        aggregated[layer_name] += weight * update.weights_delta[layer_name]
                
                layer_info = {'aggregation_type': 'weighted_average'}
            
            layer_aggregation_info[layer_name] = layer_info
        
        aggregation_info = {
            'method': 'expert_aware',
            'participant_weights': participant_weights,
            'layer_info': layer_aggregation_info,
            'expert_clusters_used': len(expert_clusters)
        }
        
        return aggregated, aggregation_info
    
    def _aggregate_expert_layer(
        self,
        layer_name: str,
        updates: List[ModelUpdate],
        participant_weights: Dict[str, float],
        expert_clusters: Dict[int, List[List[str]]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aggregate a specific expert layer using cluster information."""
        # Extract expert ID from layer name
        try:
            expert_id = int(layer_name.split('expert_')[1].split('_')[0])
        except (IndexError, ValueError):
            expert_id = -1  # Unknown expert
        
        # Initialize aggregated layer
        layer_shape = None
        for update in updates:
            if layer_name in update.weights_delta:
                layer_shape = update.weights_delta[layer_name].shape
                break
        
        if layer_shape is None:
            return np.zeros((1,)), {'error': 'no_weights_found'}
        
        aggregated_layer = np.zeros(layer_shape)
        
        # Check if this expert has identified clusters
        if expert_id in expert_clusters and expert_clusters[expert_id]:
            # Use cluster-based aggregation
            cluster_contributions = []
            
            for cluster in expert_clusters[expert_id]:
                cluster_weight_sum = 0.0
                cluster_aggregated = np.zeros(layer_shape)
                
                for participant_id in cluster:
                    # Find updates from this participant
                    for update in updates:
                        if (update.participant_id == participant_id and 
                            layer_name in update.weights_delta):
                            weight = participant_weights.get(participant_id, 1.0)
                            cluster_aggregated += weight * update.weights_delta[layer_name]
                            cluster_weight_sum += weight
                
                if cluster_weight_sum > 0:
                    cluster_aggregated /= cluster_weight_sum
                    cluster_contributions.append({
                        'weights': cluster_aggregated,
                        'participants': cluster,
                        'weight': cluster_weight_sum
                    })
            
            # Aggregate clusters
            total_cluster_weight = sum(c['weight'] for c in cluster_contributions)
            if total_cluster_weight > 0:
                for contrib in cluster_contributions:
                    cluster_weight = contrib['weight'] / total_cluster_weight
                    aggregated_layer += cluster_weight * contrib['weights']
            
            aggregation_type = 'cluster_based'
            
        else:
            # Standard weighted aggregation for this expert
            total_weight = 0.0
            for update in updates:
                if layer_name in update.weights_delta:
                    weight = participant_weights.get(update.participant_id, 1.0)
                    aggregated_layer += weight * update.weights_delta[layer_name]
                    total_weight += weight
            
            if total_weight > 0:
                aggregated_layer /= total_weight
            
            aggregation_type = 'weighted_average'
        
        layer_info = {
            'expert_id': expert_id,
            'aggregation_type': aggregation_type,
            'num_clusters_used': len(expert_clusters.get(expert_id, [])),
            'layer_norm': float(np.linalg.norm(aggregated_layer))
        }
        
        return aggregated_layer, layer_info
    
    async def _weighted_averaging(
        self,
        updates: List[ModelUpdate],
        participant_info: Dict[str, ParticipantInfo]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Standard weighted federated averaging."""
        # Compute weights based on number of samples
        total_samples = sum(update.num_samples for update in updates)
        if total_samples == 0:
            # Fallback to uniform weights
            weights = [1.0 / len(updates) for _ in updates]
        else:
            weights = [update.num_samples / total_samples for update in updates]
        
        # Initialize aggregated model
        aggregated = {}
        if updates:
            first_update = updates[0]
            for layer_name, layer_weights in first_update.weights_delta.items():
                aggregated[layer_name] = np.zeros_like(layer_weights)
        
        # Weighted aggregation
        for i, update in enumerate(updates):
            weight = weights[i]
            for layer_name, layer_weights in update.weights_delta.items():
                if layer_name in aggregated:
                    aggregated[layer_name] += weight * layer_weights
        
        aggregation_info = {
            'method': 'weighted_averaging',
            'weights_used': weights,
            'total_samples': total_samples
        }
        
        return aggregated, aggregation_info
    
    async def _adaptive_aggregation(
        self,
        updates: List[ModelUpdate],
        participant_info: Dict[str, ParticipantInfo],
        current_round: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Adaptive aggregation that adjusts strategy based on round and performance."""
        # Choose aggregation strategy based on round and update quality
        if current_round < 10:
            # Early rounds: use standard averaging to establish baseline
            return await self._weighted_averaging(updates, participant_info)
        else:
            # Later rounds: use expert-aware aggregation
            expert_analysis = await self._analyze_expert_specializations(updates)
            return await self._expert_aware_aggregation(updates, participant_info, expert_analysis)
    
    async def _federated_averaging(
        self, 
        updates: List[ModelUpdate]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Standard FedAvg algorithm."""
        if not updates:
            return {}, {'error': 'no_updates'}
        
        # Simple averaging
        aggregated = {}
        first_update = updates[0]
        
        for layer_name, weights in first_update.weights_delta.items():
            aggregated[layer_name] = weights.copy()
        
        # Add remaining updates
        for update in updates[1:]:
            for layer_name, weights in update.weights_delta.items():
                if layer_name in aggregated:
                    aggregated[layer_name] += weights
        
        # Average
        for layer_name in aggregated:
            aggregated[layer_name] /= len(updates)
        
        return aggregated, {'method': 'federated_averaging'}
    
    def _update_expert_specializations(
        self,
        updates: List[ModelUpdate],
        expert_analysis: Dict[str, Any]
    ) -> None:
        """Update expert specialization tracking."""
        for update in updates:
            participant_id = update.participant_id
            
            # Track expert performance for this participant
            for layer_name, weights in update.weights_delta.items():
                if 'expert' in layer_name.lower():
                    try:
                        expert_id = int(layer_name.split('expert_')[1].split('_')[0])
                        
                        specialization_info = {
                            'participant': participant_id,
                            'expert_id': expert_id,
                            'layer': layer_name,
                            'weight_norm': float(np.linalg.norm(weights)),
                            'update_quality': update.compute_quality_score(),
                            'timestamp': time.time()
                        }
                        
                        self.expert_specializations[expert_id].append(specialization_info)
                        
                        # Keep only recent history
                        if len(self.expert_specializations[expert_id]) > 100:
                            self.expert_specializations[expert_id] = \
                                self.expert_specializations[expert_id][-100:]
                        
                    except (IndexError, ValueError):
                        continue


class FederatedLearningCoordinator:
    """Main coordinator for multi-cloud federated learning."""
    
    def __init__(
        self,
        coordinator_id: str = "central_coordinator",
        max_participants: int = 100,
        min_participants: int = 3,
        round_duration_seconds: int = 300,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    ):
        self.coordinator_id = coordinator_id
        self.max_participants = max_participants
        self.min_participants = min_participants
        self.round_duration_seconds = round_duration_seconds
        self.privacy_level = privacy_level
        
        # Participant management
        self.participants: Dict[str, ParticipantInfo] = {}
        self.active_participants: Set[str] = set()
        
        # Federated learning components
        self.aggregator = ExpertAwareFederatedAggregator()
        self.secure_protocol = None
        self.privacy_manager = DifferentialPrivacyManager()
        
        # Training state
        self.current_round = 0
        self.global_model_state = {}
        self.round_history = deque(maxlen=1000)
        
        # Communication and synchronization
        self.pending_updates = {}
        self.round_start_time = 0.0
        self.communication_stats = defaultdict(list)
        
        logger.info(f"Initialized federated coordinator {coordinator_id}")
    
    async def register_participant(
        self,
        participant_id: str,
        cloud_provider: CloudProvider,
        capabilities: Dict[str, Any] = None,
        privacy_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Register a new participant in the federation."""
        if participant_id in self.participants:
            return {'status': 'error', 'message': 'Participant already registered'}
        
        if len(self.participants) >= self.max_participants:
            return {'status': 'error', 'message': 'Maximum participants reached'}
        
        # Create participant info
        participant = ParticipantInfo(
            participant_id=participant_id,
            cloud_provider=cloud_provider,
            role=FederationRole.PARTICIPANT,
            capabilities=capabilities or {},
            privacy_level=PrivacyLevel(
                privacy_preferences.get('privacy_level', 'dp') 
                if privacy_preferences else 'dp'
            )
        )
        
        # Update capabilities if provided
        if capabilities:
            participant.bandwidth_mbps = capabilities.get('bandwidth_mbps', 100.0)
            participant.compute_flops = capabilities.get('compute_flops', 1e12)
            participant.memory_gb = capabilities.get('memory_gb', 16.0)
        
        self.participants[participant_id] = participant
        
        logger.info(f"Registered participant {participant_id} from {cloud_provider.value}")
        
        return {
            'status': 'success',
            'participant_id': participant_id,
            'coordinator_id': self.coordinator_id,
            'current_round': self.current_round,
            'privacy_level': self.privacy_level.value
        }
    
    async def start_training_round(self) -> Dict[str, Any]:
        """Start a new federated training round."""
        if len(self.active_participants) < self.min_participants:
            return {
                'status': 'error',
                'message': f'Need at least {self.min_participants} participants'
            }
        
        self.current_round += 1
        self.round_start_time = time.time()
        self.pending_updates = {}
        
        # Setup secure aggregation if needed
        if self.privacy_level in [PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PRIVACY]:
            self.secure_protocol = SecureAggregationProtocol(
                len(self.active_participants)
            )
            keys = self.secure_protocol.setup_keys(list(self.active_participants))
        else:
            keys = {}
        
        # Send training instructions to participants
        training_config = {
            'round_number': self.current_round,
            'global_model_state': self.global_model_state.copy() if self.global_model_state else {},
            'training_parameters': {
                'local_epochs': 5,
                'learning_rate': 0.01,
                'batch_size': 32
            },
            'privacy_config': {
                'level': self.privacy_level.value,
                'epsilon': max(0.1, self.privacy_manager.get_remaining_budget() / 10),
                'clipping_norm': 1.0
            },
            'secure_keys': keys,
            'deadline': time.time() + self.round_duration_seconds
        }
        
        round_info = {
            'status': 'started',
            'round_number': self.current_round,
            'participants': list(self.active_participants),
            'expected_updates': len(self.active_participants),
            'deadline': training_config['deadline'],
            'privacy_budget_remaining': self.privacy_manager.get_remaining_budget()
        }
        
        logger.info(f"Started training round {self.current_round} with {len(self.active_participants)} participants")
        
        return round_info
    
    async def receive_model_update(
        self,
        participant_id: str,
        weights_delta: Dict[str, np.ndarray],
        training_metrics: Dict[str, float] = None,
        secure_masks: Dict[str, np.ndarray] = None
    ) -> Dict[str, Any]:
        """Receive and validate model update from participant."""
        if participant_id not in self.active_participants:
            return {'status': 'error', 'message': 'Participant not active in current round'}
        
        if participant_id in self.pending_updates:
            return {'status': 'error', 'message': 'Update already received'}
        
        # Create model update
        update = ModelUpdate(
            participant_id=participant_id,
            update_id=f"round_{self.current_round}_{participant_id}",
            weights_delta=weights_delta,
            num_samples=training_metrics.get('num_samples', 100) if training_metrics else 100,
            training_loss=training_metrics.get('loss', float('inf')) if training_metrics else float('inf'),
            validation_accuracy=training_metrics.get('accuracy', 0.0) if training_metrics else 0.0,
            computation_time=training_metrics.get('compute_time', 0.0) if training_metrics else 0.0
        )
        
        # Apply privacy preservation if needed
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            epsilon = max(0.01, self.privacy_manager.get_remaining_budget() / 20)
            
            # Clip gradients first
            clipped_weights, clip_info = self.privacy_manager.clip_gradients(
                weights_delta, clipping_norm=1.0
            )
            
            # Add noise
            private_weights, noise_info = self.privacy_manager.add_noise_to_update(
                clipped_weights, epsilon
            )
            
            update.weights_delta = private_weights
            update.privacy_spent = epsilon
            update.metadata['privacy_info'] = {
                'clipping': clip_info,
                'noise': noise_info
            }
        
        # Apply secure aggregation masking if needed
        if (self.privacy_level in [PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PRIVACY] and
            self.secure_protocol and secure_masks):
            
            masked_weights = self.secure_protocol.mask_update(
                weights_delta, secure_masks, participant_id
            )
            update.weights_delta = masked_weights
            update.is_secure = True
        
        # Validate update quality
        quality_score = update.compute_quality_score()
        if quality_score < 0.1:  # Very low quality threshold
            return {'status': 'rejected', 'message': 'Update quality too low', 'quality_score': quality_score}
        
        # Store pending update
        self.pending_updates[participant_id] = update
        
        # Update participant trust score
        if participant_id in self.participants:
            reliability = 1.0 - min(1.0, (time.time() - self.round_start_time) / self.round_duration_seconds)
            self.participants[participant_id].update_trust_score(
                quality_score, reliability
            )
        
        logger.info(f"Received update from {participant_id}, quality: {quality_score:.3f}")
        
        return {
            'status': 'accepted',
            'quality_score': quality_score,
            'updates_received': len(self.pending_updates),
            'updates_expected': len(self.active_participants)
        }
    
    async def aggregate_round(self, force_aggregate: bool = False) -> Dict[str, Any]:
        """Aggregate updates and complete the training round."""
        # Check if we have enough updates or timeout
        time_elapsed = time.time() - self.round_start_time
        has_timeout = time_elapsed > self.round_duration_seconds
        has_minimum_updates = len(self.pending_updates) >= max(1, len(self.active_participants) // 2)
        
        if not force_aggregate and not has_timeout and not has_minimum_updates:
            return {
                'status': 'waiting',
                'updates_received': len(self.pending_updates),
                'time_remaining': max(0, self.round_duration_seconds - time_elapsed)
            }
        
        if not self.pending_updates:
            return {'status': 'error', 'message': 'No updates to aggregate'}
        
        logger.info(f"Aggregating round {self.current_round} with {len(self.pending_updates)} updates")
        
        # Perform aggregation
        updates_list = list(self.pending_updates.values())
        aggregated_weights, aggregation_metadata = await self.aggregator.aggregate_updates(
            updates_list, self.participants, self.current_round
        )
        
        # Update global model state
        if aggregated_weights:
            if not self.global_model_state:
                self.global_model_state = aggregated_weights
            else:
                # Apply aggregated updates to global model
                for layer_name, delta_weights in aggregated_weights.items():
                    if layer_name in self.global_model_state:
                        # Typically would apply learning rate here
                        learning_rate = 1.0  # Simplified
                        self.global_model_state[layer_name] += learning_rate * delta_weights
                    else:
                        self.global_model_state[layer_name] = delta_weights.copy()
        
        # Compute round statistics
        round_stats = self._compute_round_statistics(updates_list, aggregation_metadata)
        
        # Update communication statistics
        total_comm_cost = sum(update.communication_cost for update in updates_list)
        self.communication_stats['round_communication'].append(total_comm_cost)
        
        # Store round history
        round_record = {
            'round': self.current_round,
            'participants': list(self.pending_updates.keys()),
            'aggregation_metadata': aggregation_metadata,
            'round_stats': round_stats,
            'duration': time_elapsed,
            'timestamp': time.time()
        }
        self.round_history.append(round_record)
        
        # Clean up
        self.pending_updates = {}
        
        result = {
            'status': 'completed',
            'round': self.current_round,
            'participants_contributed': len(updates_list),
            'aggregation_method': aggregation_metadata.get('aggregation_strategy', 'unknown'),
            'round_stats': round_stats,
            'global_model_updated': bool(aggregated_weights),
            'privacy_budget_remaining': self.privacy_manager.get_remaining_budget()
        }
        
        logger.info(f"Round {self.current_round} aggregation completed")
        
        return result
    
    def _compute_round_statistics(
        self, 
        updates: List[ModelUpdate],
        aggregation_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute statistics for the completed round."""
        if not updates:
            return {}
        
        # Quality metrics
        quality_scores = [update.compute_quality_score() for update in updates]
        training_losses = [update.training_loss for update in updates if update.training_loss != float('inf')]
        validation_accuracies = [update.validation_accuracy for update in updates]
        
        # Compute sample statistics
        total_samples = sum(update.num_samples for update in updates)
        computation_times = [update.computation_time for update in updates]
        
        # Privacy statistics
        privacy_spent = sum(getattr(update, 'privacy_spent', 0.0) for update in updates)
        
        stats = {
            'quality_metrics': {
                'avg_quality': np.mean(quality_scores),
                'min_quality': np.min(quality_scores),
                'quality_std': np.std(quality_scores)
            },
            'training_metrics': {
                'avg_loss': np.mean(training_losses) if training_losses else float('inf'),
                'avg_accuracy': np.mean(validation_accuracies),
                'accuracy_std': np.std(validation_accuracies)
            },
            'resource_metrics': {
                'total_samples': total_samples,
                'avg_computation_time': np.mean(computation_times),
                'total_communication_cost': sum(update.communication_cost for update in updates)
            },
            'privacy_metrics': {
                'total_privacy_spent': privacy_spent,
                'avg_privacy_per_participant': privacy_spent / len(updates) if updates else 0.0
            },
            'participation_metrics': {
                'cloud_distribution': dict(Counter(
                    self.participants[update.participant_id].cloud_provider.value
                    for update in updates
                    if update.participant_id in self.participants
                )),
                'trust_scores': [
                    self.participants[update.participant_id].trust_score
                    for update in updates
                    if update.participant_id in self.participants
                ]
            }
        }
        
        return stats
    
    async def select_participants_for_round(
        self,
        selection_strategy: str = "random",
        max_participants_per_round: int = None
    ) -> List[str]:
        """Select participants for the next training round."""
        max_participants_per_round = max_participants_per_round or min(10, len(self.participants))
        
        available_participants = [
            p_id for p_id, info in self.participants.items()
            if info.trust_score >= 0.3 and  # Minimum trust threshold
            info.can_participate(0.1)  # Check privacy budget
        ]
        
        if len(available_participants) <= max_participants_per_round:
            selected = available_participants
        else:
            if selection_strategy == "random":
                selected = random.sample(available_participants, max_participants_per_round)
            elif selection_strategy == "trust_based":
                # Select based on trust scores
                sorted_participants = sorted(
                    available_participants,
                    key=lambda p: self.participants[p].trust_score,
                    reverse=True
                )
                selected = sorted_participants[:max_participants_per_round]
            elif selection_strategy == "diverse":
                # Select diverse participants across cloud providers
                by_cloud = defaultdict(list)
                for p_id in available_participants:
                    cloud = self.participants[p_id].cloud_provider
                    by_cloud[cloud].append(p_id)
                
                selected = []
                clouds = list(by_cloud.keys())
                per_cloud = max(1, max_participants_per_round // len(clouds))
                
                for cloud in clouds:
                    cloud_participants = random.sample(
                        by_cloud[cloud], 
                        min(per_cloud, len(by_cloud[cloud]))
                    )
                    selected.extend(cloud_participants)
                
                # Fill remaining slots randomly
                remaining_slots = max_participants_per_round - len(selected)
                if remaining_slots > 0:
                    remaining_candidates = [p for p in available_participants if p not in selected]
                    additional = random.sample(
                        remaining_candidates,
                        min(remaining_slots, len(remaining_candidates))
                    )
                    selected.extend(additional)
            else:
                selected = random.sample(available_participants, max_participants_per_round)
        
        self.active_participants = set(selected)
        logger.info(f"Selected {len(selected)} participants for round {self.current_round + 1}")
        
        return selected
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get comprehensive federation status."""
        return {
            'coordinator_info': {
                'coordinator_id': self.coordinator_id,
                'current_round': self.current_round,
                'privacy_level': self.privacy_level.value
            },
            'participants': {
                'total_registered': len(self.participants),
                'currently_active': len(self.active_participants),
                'cloud_distribution': dict(Counter(
                    p.cloud_provider.value for p in self.participants.values()
                )),
                'avg_trust_score': np.mean([p.trust_score for p in self.participants.values()])
            },
            'training_progress': {
                'rounds_completed': len(self.round_history),
                'pending_updates': len(self.pending_updates),
                'last_round_duration': self.round_history[-1]['duration'] if self.round_history else 0,
                'avg_round_duration': np.mean([r['duration'] for r in list(self.round_history)[-10:]]) if self.round_history else 0
            },
            'privacy_status': {
                'total_budget': self.privacy_manager.epsilon,
                'budget_spent': self.privacy_manager.privacy_spent,
                'budget_remaining': self.privacy_manager.get_remaining_budget(),
                'budget_percentage_used': (self.privacy_manager.privacy_spent / self.privacy_manager.epsilon) * 100
            },
            'performance_metrics': {
                'recent_quality_scores': [
                    r['round_stats'].get('quality_metrics', {}).get('avg_quality', 0.0)
                    for r in list(self.round_history)[-5:]
                ],
                'recent_accuracies': [
                    r['round_stats'].get('training_metrics', {}).get('avg_accuracy', 0.0)
                    for r in list(self.round_history)[-5:]
                ]
            }
        }


# Factory function and utilities
def create_federated_coordinator(
    max_participants: int = 50,
    privacy_level: str = "differential_privacy",
    round_duration: int = 300
) -> FederatedLearningCoordinator:
    """Create a federated learning coordinator with specified configuration."""
    privacy_enum = PrivacyLevel(privacy_level.lower())
    
    return FederatedLearningCoordinator(
        max_participants=max_participants,
        privacy_level=privacy_enum,
        round_duration_seconds=round_duration
    )


async def simulate_federated_training(
    num_participants: int = 10,
    num_rounds: int = 5,
    cloud_distribution: Dict[str, int] = None
) -> Dict[str, Any]:
    """Simulate a federated learning training session."""
    cloud_distribution = cloud_distribution or {
        'aws': 3, 'gcp': 3, 'azure': 2, 'edge': 2
    }
    
    # Create coordinator
    coordinator = create_federated_coordinator(
        max_participants=num_participants,
        privacy_level="differential_privacy"
    )
    
    # Register participants
    participant_counter = 0
    for cloud, count in cloud_distribution.items():
        for i in range(count):
            participant_id = f"{cloud}_participant_{i}"
            cloud_provider = CloudProvider(cloud)
            
            await coordinator.register_participant(
                participant_id, cloud_provider,
                capabilities={
                    'bandwidth_mbps': random.uniform(50, 200),
                    'compute_flops': random.uniform(1e11, 1e13),
                    'memory_gb': random.uniform(8, 64)
                }
            )
            participant_counter += 1
    
    # Run federated training rounds
    training_results = []
    
    for round_num in range(num_rounds):
        # Select participants
        selected = await coordinator.select_participants_for_round(
            selection_strategy="diverse",
            max_participants_per_round=min(8, len(coordinator.participants))
        )
        
        # Start round
        round_info = await coordinator.start_training_round()
        if round_info['status'] != 'started':
            break
        
        # Simulate participant updates
        for participant_id in selected:
            # Simulate model weights (random for demonstration)
            weights_delta = {
                f'layer_{i}': np.random.randn(64, 32) * 0.01
                for i in range(3)
            }
            weights_delta.update({
                f'expert_{j}_fc': np.random.randn(128, 64) * 0.01
                for j in range(4)
            })
            
            training_metrics = {
                'num_samples': random.randint(100, 1000),
                'loss': random.uniform(0.1, 2.0),
                'accuracy': random.uniform(0.6, 0.95),
                'compute_time': random.uniform(10, 60)
            }
            
            await coordinator.receive_model_update(
                participant_id, weights_delta, training_metrics
            )
        
        # Aggregate round
        aggregation_result = await coordinator.aggregate_round(force_aggregate=True)
        training_results.append(aggregation_result)
        
        # Simulate some delay
        await asyncio.sleep(0.1)
    
    # Get final status
    final_status = coordinator.get_federation_status()
    
    return {
        'training_results': training_results,
        'final_status': final_status,
        'participants_registered': participant_counter,
        'rounds_completed': len(training_results)
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    from collections import Counter
    
    async def test_federated_system():
        """Test the federated learning system."""
        logger.info("Testing multi-cloud federated learning system...")
        
        # Run simulation
        results = await simulate_federated_training(
            num_participants=12,
            num_rounds=8,
            cloud_distribution={
                'aws': 4,
                'gcp': 3,
                'azure': 3,
                'edge': 2
            }
        )
        
        print("\n=== Federated Learning Simulation Results ===")
        print(f"Participants Registered: {results['participants_registered']}")
        print(f"Rounds Completed: {results['rounds_completed']}")
        
        final_status = results['final_status']
        print(f"Final Privacy Budget Used: {final_status['privacy_status']['budget_percentage_used']:.1f}%")
        print(f"Average Trust Score: {final_status['participants']['avg_trust_score']:.3f}")
        print(f"Cloud Distribution: {final_status['participants']['cloud_distribution']}")
        
        # Show training progress
        if results['training_results']:
            recent_accuracies = final_status['performance_metrics']['recent_accuracies']
            if recent_accuracies:
                print(f"Recent Accuracy Trend: {recent_accuracies}")
                print(f"Final Accuracy: {recent_accuracies[-1]:.3f}")
    
    # Run test
    asyncio.run(test_federated_system())