"""Unit tests for MoE analyzer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from moe_debugger.analyzer import MoEAnalyzer
from moe_debugger.models import RoutingEvent, LoadBalanceMetrics, DiagnosticResult


class TestMoEAnalyzer:
    """Test MoE analyzer functionality."""
    
    def test_initialization(self, mock_model):
        """Test analyzer initialization."""
        analyzer = MoEAnalyzer(mock_model)
        
        assert analyzer.model == mock_model
        assert analyzer.cache == {}
        assert analyzer.analysis_history == []
    
    def test_analyze_load_balance_empty_events(self, analyzer):
        """Test load balance analysis with no events."""
        result = analyzer.analyze_load_balance([])
        assert result is None
    
    def test_analyze_load_balance(self, analyzer, sample_routing_events):
        """Test load balance analysis with sample events."""
        result = analyzer.analyze_load_balance(sample_routing_events)
        
        assert isinstance(result, LoadBalanceMetrics)
        assert result.total_tokens_processed == len(sample_routing_events)
        assert len(result.expert_loads) > 0
        assert 0 <= result.fairness_index <= 1
        assert result.max_load >= result.min_load
        assert result.coefficient_of_variation >= 0
    
    def test_analyze_load_balance_with_dead_experts(self, analyzer):
        """Test load balance analysis with dead experts."""
        # Create events where expert 0 is never selected
        events = []
        for i in range(5):
            event = RoutingEvent(
                timestamp=float(i),
                layer_idx=0,
                token_position=i,
                token="test",
                expert_weights=[0.0, 0.5, 0.5, 0.0],  # Expert 0 and 3 have zero weight
                selected_experts=[1, 2],  # Only experts 1 and 2 selected
                routing_confidence=0.8,
                sequence_id="test"
            )
            events.append(event)
        
        result = analyzer.analyze_load_balance(events)
        
        assert result is not None
        assert 0 in result.dead_experts  # Expert 0 should be dead
        assert 3 in result.dead_experts  # Expert 3 should be dead
        assert result.fairness_index < 1.0  # Should show imbalance
    
    def test_detect_dead_experts(self, analyzer, sample_routing_events):
        """Test dead expert detection."""
        dead_experts = analyzer.detect_dead_experts(sample_routing_events)
        
        # In sample events, experts 1 and 3 are always selected
        # So experts 0, 2, 4, 5, 6, 7 should be considered dead
        assert isinstance(dead_experts, list)
        assert 0 in dead_experts
        assert 2 in dead_experts
        assert 1 not in dead_experts  # Should not be dead
        assert 3 not in dead_experts  # Should not be dead
    
    def test_detect_dead_experts_with_threshold(self, analyzer):
        """Test dead expert detection with custom threshold."""
        # Create events where expert 0 is activated only once
        events = []
        for i in range(10):
            if i == 0:
                selected_experts = [0, 1]  # Expert 0 selected once
            else:
                selected_experts = [1, 2]  # Experts 1 and 2 selected
            
            event = RoutingEvent(
                timestamp=float(i),
                layer_idx=0,
                token_position=i,
                token="test",
                expert_weights=[0.1, 0.4, 0.5, 0.0],
                selected_experts=selected_experts,
                routing_confidence=0.8,
                sequence_id="test"
            )
            events.append(event)
        
        # With threshold=5, expert 0 should be considered dead (activated only once)
        dead_experts = analyzer.detect_dead_experts(events, threshold=5)
        assert 0 in dead_experts
        
        # With threshold=0, expert 0 should not be considered dead
        dead_experts = analyzer.detect_dead_experts(events, threshold=0)
        assert 0 not in dead_experts
    
    def test_detect_router_collapse(self, analyzer):
        """Test router collapse detection."""
        # Create events with low entropy (router collapse)
        low_entropy_events = []
        for i in range(10):
            event = RoutingEvent(
                timestamp=float(i),
                layer_idx=0,
                token_position=i,
                token="test",
                expert_weights=[0.9, 0.1, 0.0, 0.0],  # Very low entropy
                selected_experts=[0],
                routing_confidence=0.9,
                sequence_id="test"
            )
            low_entropy_events.append(event)
        
        assert analyzer.detect_router_collapse(low_entropy_events) is True
        
        # Create events with high entropy (no collapse)
        high_entropy_events = []
        for i in range(10):
            event = RoutingEvent(
                timestamp=float(i),
                layer_idx=0,
                token_position=i,
                token="test",
                expert_weights=[0.25, 0.25, 0.25, 0.25],  # High entropy
                selected_experts=[0, 1],
                routing_confidence=0.5,
                sequence_id="test"
            )
            high_entropy_events.append(event)
        
        assert analyzer.detect_router_collapse(high_entropy_events) is False
    
    def test_compute_routing_statistics(self, analyzer, sample_routing_events):
        """Test routing statistics computation."""
        stats = analyzer.compute_routing_statistics(sample_routing_events)
        
        assert isinstance(stats, dict)
        assert "total_routing_decisions" in stats
        assert "unique_sequences" in stats
        assert "layers_analyzed" in stats
        assert "avg_confidence" in stats
        assert "expert_usage_distribution" in stats
        assert "avg_experts_per_token" in stats
        
        assert stats["total_routing_decisions"] == len(sample_routing_events)
        assert stats["unique_sequences"] >= 1
        assert stats["layers_analyzed"] >= 1
        assert 0 <= stats["avg_confidence"] <= 1
        assert stats["avg_experts_per_token"] > 0
    
    def test_compute_expert_utilization(self, analyzer, sample_routing_events):
        """Test expert utilization computation."""
        utilization = analyzer.compute_expert_utilization(sample_routing_events)
        
        assert isinstance(utilization, dict)
        
        # Check that utilization rates are between 0 and 1
        for expert_id, rate in utilization.items():
            assert 0 <= rate <= 1
        
        # Check that experts 1 and 3 have high utilization (they're always selected)
        assert utilization.get(1, 0) > 0
        assert utilization.get(3, 0) > 0
    
    def test_analyze_token_attribution(self, analyzer, sample_routing_events):
        """Test token attribution analysis."""
        token_texts = [f"token_{i}" for i in range(len(sample_routing_events))]
        
        attributions = analyzer.analyze_token_attribution(
            sample_routing_events, token_texts
        )
        
        assert len(attributions) == len(sample_routing_events)
        
        for i, attribution in enumerate(attributions):
            assert attribution.token == token_texts[i]
            assert attribution.position == i
            assert isinstance(attribution.expert_contributions, dict)
            assert attribution.sequence_id == sample_routing_events[i].sequence_id
    
    def test_analyze_routing_entropy(self, analyzer, sample_routing_events):
        """Test routing entropy analysis."""
        entropy_stats = analyzer.analyze_routing_entropy(sample_routing_events)
        
        assert isinstance(entropy_stats, dict)
        assert "mean_entropy" in entropy_stats
        assert "std_entropy" in entropy_stats
        assert "min_entropy" in entropy_stats
        assert "max_entropy" in entropy_stats
        assert "entropy_trend" in entropy_stats
        assert "entropy_history" in entropy_stats
        
        assert entropy_stats["mean_entropy"] >= 0
        assert entropy_stats["std_entropy"] >= 0
        assert entropy_stats["min_entropy"] <= entropy_stats["max_entropy"]
    
    def test_detect_anomalies(self, analyzer, sample_routing_events):
        """Test anomaly detection."""
        diagnostics = analyzer.detect_anomalies(sample_routing_events)
        
        assert isinstance(diagnostics, list)
        
        for diagnostic in diagnostics:
            assert isinstance(diagnostic, DiagnosticResult)
            assert diagnostic.diagnostic_type in ["router_collapse", "dead_experts", "load_imbalance"]
            assert diagnostic.severity in ["info", "warning", "error", "critical"]
            assert isinstance(diagnostic.message, str)
            assert isinstance(diagnostic.affected_experts, list)
            assert isinstance(diagnostic.suggested_actions, list)
            assert isinstance(diagnostic.metrics, dict)
    
    def test_generate_optimization_suggestions(self, analyzer, sample_routing_events):
        """Test optimization suggestion generation."""
        suggestions = analyzer.generate_optimization_suggestions(sample_routing_events)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
    
    def test_generate_optimization_suggestions_empty_events(self, analyzer):
        """Test optimization suggestions with no events."""
        suggestions = analyzer.generate_optimization_suggestions([])
        
        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        assert "No routing data available" in suggestions[0]
    
    @patch('moe_debugger.analyzer.MoEAnalyzer._extract_expert_parameters')
    def test_compute_expert_similarity(self, mock_extract, analyzer, mock_model):
        """Test expert similarity computation."""
        # Mock expert parameters
        mock_extract.return_value = {
            0: [torch.randn(100, 50), torch.randn(50)],
            1: [torch.randn(100, 50), torch.randn(50)],
            2: [torch.randn(100, 50), torch.randn(50)]
        }
        
        similarities = analyzer.compute_expert_similarity()
        
        assert isinstance(similarities, dict)
        
        # Check that we have pairwise similarities
        expected_pairs = [(0, 1), (0, 2), (1, 2)]
        for pair in expected_pairs:
            assert pair in similarities
            # Cosine similarity should be between -1 and 1
            assert -1 <= similarities[pair] <= 1
    
    def test_clear_cache(self, analyzer):
        """Test cache clearing."""
        # Add some data to cache
        analyzer.cache["test_key"] = "test_value"
        analyzer.analysis_history.append("test_analysis")
        
        analyzer.clear_cache()
        
        assert analyzer.cache == {}
        assert analyzer.analysis_history == []


class TestAnalyzerEdgeCases:
    """Test analyzer edge cases and error handling."""
    
    def test_empty_events_handling(self, analyzer):
        """Test handling of empty event lists."""
        assert analyzer.analyze_load_balance([]) is None
        assert analyzer.detect_dead_experts([]) == []
        assert analyzer.detect_router_collapse([]) is False
        assert analyzer.compute_routing_statistics([]) == {}
        assert analyzer.compute_expert_utilization([]) == {}
        assert analyzer.analyze_token_attribution([]) == []
        assert analyzer.analyze_routing_entropy([]) == {}
    
    def test_single_event_handling(self, analyzer):
        """Test handling of single routing event."""
        event = RoutingEvent(
            timestamp=1.0,
            layer_idx=0,
            token_position=0,
            token="test",
            expert_weights=[0.5, 0.5],
            selected_experts=[0, 1],
            routing_confidence=0.7,
            sequence_id="test"
        )
        
        result = analyzer.analyze_load_balance([event])
        assert result is not None
        assert result.total_tokens_processed == 1
        
        stats = analyzer.compute_routing_statistics([event])
        assert stats["total_routing_decisions"] == 1
        
        utilization = analyzer.compute_expert_utilization([event])
        assert utilization[0] == 1.0
        assert utilization[1] == 1.0
    
    def test_invalid_expert_weights(self, analyzer):
        """Test handling of invalid expert weights."""
        # Empty weights
        event = RoutingEvent(
            timestamp=1.0,
            layer_idx=0,
            token_position=0,
            token="test",
            expert_weights=[],
            selected_experts=[],
            routing_confidence=0.0,
            sequence_id="test"
        )
        
        # Should not crash
        entropy_stats = analyzer.analyze_routing_entropy([event])
        assert isinstance(entropy_stats, dict)