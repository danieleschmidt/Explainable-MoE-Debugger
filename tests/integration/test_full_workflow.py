"""Integration tests for full debugging workflow."""

import pytest
import torch
import time
from unittest.mock import Mock, patch

from moe_debugger import MoEDebugger, MoEAnalyzer, MoEProfiler
from moe_debugger.models import RoutingEvent


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete debugging workflow."""
    
    def test_complete_debugging_session(self, mock_model):
        """Test a complete debugging session from start to finish."""
        # Initialize debugger
        debugger = MoEDebugger(mock_model, {
            "sampling_rate": 1.0,
            "buffer_size": 1000,
            "enabled_hooks": {"router": True, "experts": True}
        })
        
        # Start session
        session = debugger.start_session("integration_test")
        assert session is not None
        assert debugger.is_active is True
        
        # Simulate model inference with tracing
        input_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
        
        with debugger.trace("test_inference"):
            # Mock some routing events
            for i in range(10):
                event = RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=0,
                    token_position=i,
                    token=f"token_{i}",
                    expert_weights=[0.1, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0],
                    selected_experts=[1, 2],
                    routing_confidence=0.7 + (i * 0.01),
                    sequence_id="test_inference"
                )
                debugger.hooks_manager.routing_events.append(event)
        
        # Get routing statistics
        routing_stats = debugger.get_routing_stats()
        assert routing_stats is not None
        assert "total_routing_decisions" in routing_stats
        assert routing_stats["total_routing_decisions"] == 10
        
        # Get expert utilization
        utilization = debugger.get_expert_utilization()
        assert isinstance(utilization, dict)
        assert 1 in utilization  # Expert 1 was always selected
        assert 2 in utilization  # Expert 2 was always selected
        
        # Detect issues
        issues = debugger.detect_issues()
        assert isinstance(issues, list)
        # Should detect dead experts (experts 0, 3, 4, 5, 6, 7)
        dead_expert_issues = [issue for issue in issues if issue["type"] == "dead_experts"]
        assert len(dead_expert_issues) > 0
        
        # Get performance metrics
        performance = debugger.get_performance_metrics()
        assert isinstance(performance, dict)
        
        # Get model summary
        summary = debugger.get_model_summary()
        assert isinstance(summary, dict)
        assert summary["is_active"] is True
        assert summary["session_id"] == "integration_test"
        
        # End session
        ended_session = debugger.end_session()
        assert ended_session == session
        assert debugger.is_active is False
        assert ended_session.end_time is not None
    
    def test_multi_layer_analysis(self, mock_model):
        """Test analysis across multiple layers."""
        debugger = MoEDebugger(mock_model, {"sampling_rate": 1.0})
        
        with debugger:
            # Add events for multiple layers
            for layer_idx in range(2):
                for token_pos in range(5):
                    event = RoutingEvent(
                        timestamp=time.time(),
                        layer_idx=layer_idx,
                        token_position=token_pos,
                        token=f"token_{token_pos}",
                        expert_weights=[0.2, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                        selected_experts=[1, 2],
                        routing_confidence=0.6,
                        sequence_id=f"layer_{layer_idx}_seq"
                    )
                    debugger.hooks_manager.routing_events.append(event)
            
            # Analyze routing statistics
            stats = debugger.get_routing_stats()
            assert stats["total_routing_decisions"] == 10
            assert stats["layers_analyzed"] == 2
            assert stats["unique_sequences"] == 2
    
    def test_load_balancing_analysis(self, mock_model):
        """Test comprehensive load balancing analysis."""
        analyzer = MoEAnalyzer(mock_model)
        
        # Create events with imbalanced expert usage
        events = []
        expert_usage = [50, 30, 15, 5, 0, 0, 0, 0]  # Highly imbalanced
        
        for expert_id, usage_count in enumerate(expert_usage):
            for _ in range(usage_count):
                event = RoutingEvent(
                    timestamp=time.time(),
                    layer_idx=0,
                    token_position=len(events),
                    token=f"token_{len(events)}",
                    expert_weights=[0.0] * 8,  # Will be set based on expert_id
                    selected_experts=[expert_id],
                    routing_confidence=0.8,
                    sequence_id="load_balance_test"
                )
                # Set weight for selected expert
                event.expert_weights[expert_id] = 1.0
                events.append(event)
        
        # Analyze load balance
        load_metrics = analyzer.analyze_load_balance(events)
        
        assert load_metrics is not None
        assert load_metrics.total_tokens_processed == 100
        assert len(load_metrics.dead_experts) == 3  # Experts 5, 6, 7
        assert load_metrics.fairness_index < 0.8  # Should show imbalance
        assert load_metrics.max_load == 50
        assert load_metrics.min_load == 0
        
        # Check that overloaded experts are detected
        assert len(load_metrics.overloaded_experts) > 0
        assert 0 in load_metrics.overloaded_experts  # Expert 0 has 50 tokens (high load)
    
    def test_router_collapse_detection(self, mock_model):
        """Test router collapse detection."""
        analyzer = MoEAnalyzer(mock_model)
        
        # Create events showing router collapse (low entropy)
        collapsed_events = []
        for i in range(20):
            event = RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Very low entropy
                selected_experts=[0],
                routing_confidence=0.95,
                sequence_id="collapsed"
            )
            collapsed_events.append(event)
        
        # Should detect collapse
        assert analyzer.detect_router_collapse(collapsed_events) is True
        
        # Test with healthy routing (high entropy)
        healthy_events = []
        for i in range(20):
            event = RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],  # High entropy
                selected_experts=[0, 1],
                routing_confidence=0.5,
                sequence_id="healthy"
            )
            healthy_events.append(event)
        
        # Should not detect collapse
        assert analyzer.detect_router_collapse(healthy_events) is False
    
    def test_performance_profiling_integration(self, mock_model):
        """Test performance profiling integration."""
        profiler = MoEProfiler(mock_model)
        
        # Start profiling
        profiler.start_profiling()
        assert profiler.is_profiling is True
        
        # Simulate some work with profiling contexts
        with profiler.profile_inference():
            time.sleep(0.01)  # Simulate inference time
            
            with profiler.profile_layer("layer_0"):
                time.sleep(0.005)
                
                with profiler.profile_expert(0, 0):
                    time.sleep(0.002)
                
                with profiler.profile_expert(1, 0):
                    time.sleep(0.003)
            
            with profiler.profile_routing(0):
                time.sleep(0.001)
        
        # Record some metrics
        profiler.record_tokens_processed(10)
        profiler.record_cache_hit()
        profiler.record_cache_hit()
        profiler.record_cache_miss()
        
        # Get current metrics
        metrics = profiler.get_current_metrics()
        
        assert metrics["is_profiling"] is True
        assert metrics["tokens_processed"] == 10
        assert metrics["cache_hit_rate"] == 2/3  # 2 hits, 1 miss
        assert "avg_layer_times_ms" in metrics
        assert "avg_expert_times_ms" in metrics
        
        # Stop profiling
        profiler.stop_profiling()
        assert profiler.is_profiling is False
        
        # Check that a profile was created
        profiles = profiler.get_profiles()
        assert len(profiles) == 1
        
        profile = profiles[0]
        assert profile.total_inference_time_ms > 0
        assert profile.token_throughput > 0
        assert profile.cache_hit_rate == 2/3
    
    def test_anomaly_detection_workflow(self, mock_model):
        """Test complete anomaly detection workflow."""
        analyzer = MoEAnalyzer(mock_model)
        
        # Create problematic routing events
        events = []
        
        # Add events showing multiple issues:
        # 1. Dead experts (experts 4-7 never used)
        # 2. Router collapse (low entropy)
        # 3. Load imbalance
        
        for i in range(50):
            if i < 40:
                # Most tokens go to expert 0 (collapse + imbalance)
                expert_weights = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                selected_experts = [0]
            else:
                # Some tokens go to expert 1
                expert_weights = [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                selected_experts = [1]
            
            event = RoutingEvent(
                timestamp=time.time() + i * 0.001,
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=expert_weights,
                selected_experts=selected_experts,
                routing_confidence=0.9,
                sequence_id="anomaly_test"
            )
            events.append(event)
        
        # Detect anomalies
        diagnostics = analyzer.detect_anomalies(events)
        
        # Should detect multiple issues
        diagnostic_types = {d.diagnostic_type for d in diagnostics}
        
        assert "dead_experts" in diagnostic_types
        assert "router_collapse" in diagnostic_types
        assert "load_imbalance" in diagnostic_types
        
        # Check dead experts diagnostic
        dead_expert_diagnostic = next(d for d in diagnostics if d.diagnostic_type == "dead_experts")
        assert len(dead_expert_diagnostic.affected_experts) >= 4  # Experts 4-7 at minimum
        assert "increase expert capacity" in " ".join(dead_expert_diagnostic.suggested_actions).lower()
        
        # Check router collapse diagnostic
        collapse_diagnostic = next(d for d in diagnostics if d.diagnostic_type == "router_collapse")
        assert collapse_diagnostic.severity == "critical"
        assert "temperature" in " ".join(collapse_diagnostic.suggested_actions).lower()
        
        # Check load imbalance diagnostic
        balance_diagnostic = next(d for d in diagnostics if d.diagnostic_type == "load_imbalance")
        assert balance_diagnostic.severity == "warning"
        assert "load balancing" in " ".join(balance_diagnostic.suggested_actions).lower()
    
    def test_optimization_suggestions(self, mock_model):
        """Test optimization suggestion generation."""
        analyzer = MoEAnalyzer(mock_model)
        
        # Create suboptimal routing events
        events = []
        for i in range(30):
            event = RoutingEvent(
                timestamp=time.time(),
                layer_idx=0,
                token_position=i,
                token=f"token_{i}",
                expert_weights=[0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Low entropy
                selected_experts=[0],  # Only one expert selected
                routing_confidence=0.8,
                sequence_id="optimization_test"
            )
            events.append(event)
        
        # Generate suggestions
        suggestions = analyzer.generate_optimization_suggestions(events)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest improvements
        suggestion_text = " ".join(suggestions).lower()
        
        # Should mention dead experts
        assert "dead expert" in suggestion_text or "reviving" in suggestion_text
        
        # Should mention entropy or temperature
        assert "entropy" in suggestion_text or "temperature" in suggestion_text
        
        # Should mention experts per token
        assert "expert" in suggestion_text and ("capacity" in suggestion_text or "top-k" in suggestion_text)
    
    @pytest.mark.slow
    def test_large_scale_analysis(self, mock_model):
        """Test analysis with large number of events."""
        analyzer = MoEAnalyzer(mock_model)
        
        # Generate large number of events
        events = []
        for i in range(1000):
            # Simulate realistic expert selection
            if i % 10 == 0:
                selected_experts = [0, 1]
                weights = [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif i % 10 < 7:
                selected_experts = [1, 2]
                weights = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                selected_experts = [2, 3]
                weights = [0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0]
            
            event = RoutingEvent(
                timestamp=time.time() + i * 0.001,
                layer_idx=i % 3,  # Multiple layers
                token_position=i % 50,
                token=f"token_{i % 100}",
                expert_weights=weights,
                selected_experts=selected_experts,
                routing_confidence=0.6 + (i % 40) * 0.01,
                sequence_id=f"seq_{i // 50}"
            )
            events.append(event)
        
        # Perform various analyses
        start_time = time.time()
        
        load_metrics = analyzer.analyze_load_balance(events)
        routing_stats = analyzer.compute_routing_statistics(events)
        entropy_stats = analyzer.analyze_routing_entropy(events)
        dead_experts = analyzer.detect_dead_experts(events)
        
        analysis_time = time.time() - start_time
        
        # Verify results
        assert load_metrics is not None
        assert load_metrics.total_tokens_processed == 1000
        
        assert routing_stats["total_routing_decisions"] == 1000
        assert routing_stats["layers_analyzed"] == 3
        assert routing_stats["unique_sequences"] == 20
        
        assert isinstance(entropy_stats, dict)
        assert entropy_stats["mean_entropy"] > 0
        
        # Dead experts should include 4, 5, 6, 7
        assert len(dead_experts) >= 4
        
        # Analysis should complete in reasonable time (< 1 second for 1000 events)
        assert analysis_time < 1.0