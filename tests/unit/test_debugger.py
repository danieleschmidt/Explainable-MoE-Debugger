"""Unit tests for MoE debugger."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from moe_debugger.debugger import MoEDebugger
from moe_debugger.models import DebugSession, HookConfiguration


class TestMoEDebugger:
    """Test MoE debugger functionality."""
    
    def test_initialization(self, mock_model):
        """Test debugger initialization."""
        debugger = MoEDebugger(mock_model)
        
        assert debugger.model == mock_model
        assert isinstance(debugger.config, HookConfiguration)
        assert debugger.hooks_manager is not None
        assert debugger.analyzer is not None
        assert debugger.profiler is not None
        assert debugger.current_session is None
        assert debugger.is_active is False
        assert debugger.architecture is not None
    
    def test_initialization_with_config(self, mock_model):
        """Test debugger initialization with custom config."""
        config = {
            "sampling_rate": 0.5,
            "buffer_size": 5000,
            "enabled_hooks": {"router": True, "experts": False}
        }
        
        debugger = MoEDebugger(mock_model, config)
        
        assert debugger.config.sampling_rate == 0.5
        assert debugger.config.buffer_size == 5000
        assert debugger.config.enabled_hooks["router"] is True
        assert debugger.config.enabled_hooks["experts"] is False
    
    def test_create_config(self, mock_model):
        """Test configuration creation."""
        debugger = MoEDebugger(mock_model)
        
        user_config = {"sampling_rate": 0.8, "custom_param": "test"}
        config = debugger._create_config(user_config)
        
        assert isinstance(config, HookConfiguration)
        assert config.sampling_rate == 0.8
        # Should merge with defaults
        assert config.buffer_size == 10000  # Default value
        assert config.save_gradients is False  # Default value
    
    def test_analyze_architecture(self, mock_model):
        """Test model architecture analysis."""
        debugger = MoEDebugger(mock_model)
        arch = debugger.architecture
        
        assert arch.num_layers == mock_model.num_layers
        assert arch.num_experts_per_layer == mock_model.num_experts
        assert arch.hidden_size == mock_model.hidden_size
        assert arch.intermediate_size == mock_model.hidden_size * 4
        assert arch.expert_capacity == 2.0  # Default
    
    def test_start_session(self, debugger):
        """Test starting a debug session."""
        assert debugger.current_session is None
        assert debugger.is_active is False
        
        session = debugger.start_session("test_session")
        
        assert isinstance(session, DebugSession)
        assert session.session_id == "test_session"
        assert session.model_name == debugger.model.__class__.__name__
        assert debugger.current_session == session
        assert debugger.is_active is True
    
    def test_start_session_auto_id(self, debugger):
        """Test starting session with auto-generated ID."""
        session = debugger.start_session()
        
        assert session.session_id.startswith("session_")
        assert debugger.current_session == session
    
    def test_end_session(self, debugger):
        """Test ending a debug session."""
        # Start a session first
        session = debugger.start_session("test_session")
        assert debugger.is_active is True
        
        # End the session
        ended_session = debugger.end_session()
        
        assert ended_session == session
        assert ended_session.end_time is not None
        assert debugger.current_session is None
        assert debugger.is_active is False
    
    def test_end_session_no_active(self, debugger):
        """Test ending session when none is active."""
        result = debugger.end_session()
        assert result is None
    
    def test_start_session_ends_previous(self, debugger):
        """Test that starting new session ends the previous one."""
        # Start first session
        session1 = debugger.start_session("session1")
        assert debugger.current_session == session1
        
        # Start second session
        session2 = debugger.start_session("session2")
        assert debugger.current_session == session2
        assert debugger.current_session != session1
    
    @patch('moe_debugger.debugger.MoEDebugger._emit_event')
    def test_session_callbacks(self, mock_emit, debugger):
        """Test that session events trigger callbacks."""
        # Start session
        session = debugger.start_session("test_session")
        mock_emit.assert_called_with("session_start", session)
        
        mock_emit.reset_mock()
        
        # End session
        ended_session = debugger.end_session()
        mock_emit.assert_called_with("session_end", ended_session)
    
    def test_trace_context_manager(self, debugger):
        """Test trace context manager."""
        # Start session first
        debugger.start_session("test_session")
        
        with debugger.trace("test_trace") as dbg:
            assert dbg == debugger
            # Check that sequence was started
            assert debugger.hooks_manager.current_sequence_id == "test_trace"
        
        # Check that sequence was ended
        assert debugger.hooks_manager.current_sequence_id is None
    
    def test_trace_auto_start_session(self, debugger):
        """Test that trace starts session if none active."""
        assert debugger.is_active is False
        
        with debugger.trace() as dbg:
            assert debugger.is_active is True
    
    def test_register_callback(self, debugger):
        """Test callback registration."""
        callback = Mock()
        
        debugger.register_callback("session_start", callback)
        
        # Check callback was registered
        assert callback in debugger.event_callbacks["session_start"]
    
    def test_emit_event(self, debugger):
        """Test event emission."""
        callback1 = Mock()
        callback2 = Mock()
        
        debugger.register_callback("test_event", callback1)
        debugger.register_callback("test_event", callback2)
        
        test_data = {"test": "data"}
        debugger._emit_event("test_event", test_data)
        
        callback1.assert_called_once_with(test_data)
        callback2.assert_called_once_with(test_data)
    
    def test_emit_event_callback_error(self, debugger):
        """Test that callback errors don't crash event emission."""
        failing_callback = Mock(side_effect=Exception("Test error"))
        working_callback = Mock()
        
        debugger.register_callback("test_event", failing_callback)
        debugger.register_callback("test_event", working_callback)
        
        # Should not raise exception
        debugger._emit_event("test_event", {"test": "data"})
        
        # Working callback should still be called
        working_callback.assert_called_once()
    
    @patch('moe_debugger.debugger.MoEDebugger.hooks_manager')
    def test_get_routing_stats(self, mock_hooks, debugger):
        """Test getting routing statistics."""
        mock_hooks.get_routing_events.return_value = []
        debugger.analyzer.compute_routing_statistics = Mock(return_value={"test": "stats"})
        
        # Start session
        debugger.start_session("test")
        
        stats = debugger.get_routing_stats()
        
        assert stats == {"test": "stats"}
        debugger.analyzer.compute_routing_statistics.assert_called_once()
    
    def test_get_routing_stats_no_session(self, debugger):
        """Test getting routing stats with no active session."""
        stats = debugger.get_routing_stats()
        assert stats == {}
    
    @patch('moe_debugger.debugger.MoEDebugger.hooks_manager')
    def test_get_expert_utilization(self, mock_hooks, debugger):
        """Test getting expert utilization."""
        mock_hooks.get_routing_events.return_value = []
        debugger.analyzer.compute_expert_utilization = Mock(return_value={"expert_0": 0.5})
        
        utilization = debugger.get_expert_utilization()
        
        assert utilization == {"expert_0": 0.5}
        debugger.analyzer.compute_expert_utilization.assert_called_once()
    
    @patch('moe_debugger.debugger.MoEDebugger.profiler')
    def test_get_performance_metrics(self, mock_profiler, debugger):
        """Test getting performance metrics."""
        mock_profiler.get_current_metrics.return_value = {"memory": 100}
        
        metrics = debugger.get_performance_metrics()
        
        assert metrics == {"memory": 100}
        mock_profiler.get_current_metrics.assert_called_once()
    
    @patch('moe_debugger.debugger.MoEDebugger.hooks_manager')
    @patch('moe_debugger.debugger.MoEDebugger.analyzer')
    def test_detect_issues(self, mock_analyzer, mock_hooks, debugger):
        """Test issue detection."""
        # Setup mocks
        mock_hooks.get_routing_events.return_value = []
        mock_analyzer.detect_dead_experts.return_value = [2, 5]
        mock_analyzer.detect_router_collapse.return_value = True
        mock_analyzer.analyze_load_balance.return_value = Mock(fairness_index=0.5)
        
        # Start session
        debugger.start_session("test")
        
        issues = debugger.detect_issues()
        
        assert isinstance(issues, list)
        assert len(issues) == 3  # dead experts, router collapse, load imbalance
        
        # Check dead experts issue
        dead_expert_issue = next(issue for issue in issues if issue["type"] == "dead_experts")
        assert dead_expert_issue["severity"] == "warning"
        assert "2 dead experts" in dead_expert_issue["message"]
        
        # Check router collapse issue
        collapse_issue = next(issue for issue in issues if issue["type"] == "router_collapse")
        assert collapse_issue["severity"] == "critical"
        
        # Check load imbalance issue
        balance_issue = next(issue for issue in issues if issue["type"] == "load_imbalance")
        assert balance_issue["severity"] == "warning"
    
    def test_detect_issues_no_session(self, debugger):
        """Test issue detection with no active session."""
        issues = debugger.detect_issues()
        assert issues == []
    
    def test_set_sampling_rate(self, debugger):
        """Test setting sampling rate."""
        debugger.set_sampling_rate(0.75)
        
        assert debugger.config.sampling_rate == 0.75
        assert debugger.hooks_manager.config.sampling_rate == 0.75
    
    def test_set_sampling_rate_bounds(self, debugger):
        """Test sampling rate bounds checking."""
        # Test lower bound
        debugger.set_sampling_rate(-0.5)
        assert debugger.config.sampling_rate == 0.0
        
        # Test upper bound
        debugger.set_sampling_rate(1.5)
        assert debugger.config.sampling_rate == 1.0
    
    def test_clear_data(self, debugger):
        """Test clearing debugger data."""
        debugger.hooks_manager.clear_data = Mock()
        debugger.profiler.clear_data = Mock()
        debugger.analyzer.clear_cache = Mock()
        
        debugger.clear_data()
        
        debugger.hooks_manager.clear_data.assert_called_once()
        debugger.profiler.clear_data.assert_called_once()
        debugger.analyzer.clear_cache.assert_called_once()
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_export_session(self, mock_json_dump, mock_open, debugger):
        """Test session export."""
        # Start session with some data
        session = debugger.start_session("test_session")
        session.routing_events = [Mock()]  # Add some mock data
        
        debugger.export_session("test_export.json")
        
        mock_open.assert_called_once_with("test_export.json", 'w')
        mock_json_dump.assert_called_once()
        
        # Check that session data was prepared for export
        call_args = mock_json_dump.call_args[0]
        session_data = call_args[0]
        
        assert session_data["session_id"] == "test_session"
        assert session_data["model_name"] == session.model_name
        assert "routing_events" in session_data
        assert "architecture" in session_data
    
    def test_export_session_no_active(self, debugger):
        """Test export session with no active session."""
        with pytest.raises(ValueError, match="No active session to export"):
            debugger.export_session("test.json")
    
    def test_get_model_summary(self, debugger):
        """Test getting model summary."""
        summary = debugger.get_model_summary()
        
        assert isinstance(summary, dict)
        assert "architecture" in summary
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "model_class" in summary
        assert "device" in summary
        assert "dtype" in summary
        assert "is_active" in summary
        assert "session_id" in summary
        
        assert summary["model_class"] == debugger.model.__class__.__name__
        assert summary["is_active"] == debugger.is_active
    
    def test_context_manager(self, debugger):
        """Test debugger as context manager."""
        assert debugger.is_active is False
        
        with debugger as dbg:
            assert dbg == debugger
            assert debugger.is_active is True
        
        assert debugger.is_active is False
    
    def test_force_expert_selection_not_implemented(self, debugger):
        """Test that force expert selection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            debugger.force_expert_selection([0, 1, 2])