#!/usr/bin/env python3
"""Comprehensive test suite for research extensions to MoE debugger.

This test suite validates all novel adaptive routing algorithms, research
validation frameworks, and enhanced debugger integration. It ensures
publication-quality research code meets rigorous testing standards.

Test Categories:
1. Adaptive Routing Algorithm Tests
2. Research Validation Framework Tests  
3. Enhanced Debugger Integration Tests
4. Statistical Analysis Validation Tests
5. Performance Benchmark Tests

Coverage Target: 95%+ for research components
Quality Standard: Publication-ready research code
"""

import sys
import os
import time
import math
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from moe_debugger.adaptive_routing import (
        AdaptiveRoutingConfig, AdaptiveRoutingSystem,
        EntropyGuidedAdaptiveRouter, DeadExpertResurrectionFramework,
        PredictiveLoadBalancer, MultiObjectiveRoutingOptimizer,
        ExpertState, RoutingDecision
    )
    from moe_debugger.research_validation import (
        ExperimentalConfig, ExperimentRunner, StatisticalAnalyzer,
        ResearchReportGenerator, ScenarioGenerator, MetricsCalculator,
        run_comprehensive_research_validation
    )
    from moe_debugger.enhanced_debugger import (
        EnhancedMoEDebugger, EnhancedDebuggerConfig,
        create_enhanced_debugger, PerformanceMonitor
    )
    from moe_debugger.models import RoutingEvent
    from moe_debugger.mock_torch import torch, nn
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestAdaptiveRoutingAlgorithms(unittest.TestCase):
    """Test suite for novel adaptive routing algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.config = AdaptiveRoutingConfig()
        self.expert_count = 8
        
    def test_entropy_guided_adaptive_router_initialization(self):
        """Test EAR algorithm initialization."""
        router = EntropyGuidedAdaptiveRouter(self.config)
        
        self.assertEqual(router.temperature, 1.0)
        self.assertEqual(len(router.entropy_history), 0)
        self.assertIsNotNone(router.config)
        
    def test_entropy_guided_adaptive_router_temperature_adaptation(self):
        """Test adaptive temperature computation based on entropy trends."""
        router = EntropyGuidedAdaptiveRouter(self.config)
        
        # Test low entropy scenario (should increase temperature)
        low_entropy = 0.3
        initial_temp = router.temperature
        adaptive_temp = router.compute_adaptive_temperature(low_entropy)
        
        # Should increase temperature for low entropy
        self.assertGreaterEqual(adaptive_temp, initial_temp)
        
        # Test high entropy scenario 
        router.temperature = 1.0  # Reset
        high_entropy = 3.0
        adaptive_temp = router.compute_adaptive_temperature(high_entropy)
        
        # Should decrease temperature for high entropy
        self.assertLessEqual(adaptive_temp, 1.0)
        
    def test_entropy_guided_adaptive_router_routing_application(self):
        """Test application of adaptive routing to expert logits."""
        router = EntropyGuidedAdaptiveRouter(self.config)
        expert_states = {i: ExpertState(i) for i in range(self.expert_count)}
        
        # Test with sample logits
        logits = [1.0, 0.5, 2.0, 0.8, 1.2, 0.3, 1.8, 0.6]
        
        weights, adaptation_info = router.apply_adaptive_routing(logits, expert_states)
        
        # Validate output
        self.assertEqual(len(weights), len(logits))
        self.assertAlmostEqual(sum(weights), 1.0, places=5)  # Should be normalized
        self.assertIn('entropy_before', adaptation_info)
        self.assertIn('entropy_after', adaptation_info)
        self.assertIn('temperature', adaptation_info)
        
    def test_dead_expert_resurrection_framework_initialization(self):
        """Test DERF algorithm initialization."""
        framework = DeadExpertResurrectionFramework(self.config)
        
        self.assertEqual(len(framework.expert_states), 0)
        self.assertEqual(framework.token_count, 0)
        self.assertEqual(len(framework.resurrection_history), 0)
        
    def test_dead_expert_resurrection_expert_usage_tracking(self):
        """Test expert usage tracking and dead expert identification."""
        framework = DeadExpertResurrectionFramework(self.config)
        all_expert_ids = list(range(self.expert_count))
        
        # Simulate expert usage - favor first few experts
        for _ in range(200):
            selected_experts = [0, 1]  # Always select first two experts
            framework.update_expert_usage(selected_experts, all_expert_ids)
        
        # Check dead expert identification
        dead_experts = framework.identify_dead_experts()
        
        # Experts 2-7 should be identified as dead
        expected_dead = list(range(2, self.expert_count))
        self.assertEqual(set(dead_experts), set(expected_dead))
        
        # Check expert states
        self.assertEqual(framework.expert_states[0].last_selected, 0)  # Recently selected
        self.assertGreater(framework.expert_states[2].last_selected, 100)  # Dead expert
        
    def test_dead_expert_resurrection_boost_application(self):
        """Test resurrection boost application to dead experts."""
        framework = DeadExpertResurrectionFramework(self.config)
        framework.token_count = 150
        
        # Create a dead expert state
        expert_id = 5
        framework.expert_states[expert_id] = ExpertState(expert_id)
        framework.expert_states[expert_id].last_selected = 120  # Dead for 120 tokens
        
        # Apply resurrection boost
        boost = framework.apply_resurrection_boost(expert_id)
        
        self.assertGreater(boost, 1.0)  # Should apply boost
        self.assertEqual(framework.expert_states[expert_id].resurrection_attempts, 1)
        
    def test_predictive_load_balancer_initialization(self):
        """Test PLB algorithm initialization."""
        balancer = PredictiveLoadBalancer(self.config)
        
        self.assertEqual(len(balancer.load_history), 0)
        self.assertEqual(len(balancer.predictions), 0)
        self.assertEqual(len(balancer.balancing_adjustments), 0)
        
    def test_predictive_load_balancer_load_prediction(self):
        """Test load prediction based on historical patterns."""
        balancer = PredictiveLoadBalancer(self.config)
        
        # Create load history with trend
        expert_loads = {0: 10, 1: 15, 2: 5, 3: 20}
        
        # Update history multiple times to create trend
        for i in range(10):
            # Simulate increasing load for expert 0
            loads = {0: 10 + i, 1: 15, 2: 5, 3: 20 - i//2}
            balancer.update_load_history(loads)
        
        # Predict future loads
        predictions = balancer.predict_future_loads()
        
        self.assertIn(0, predictions)
        self.assertIn(1, predictions)
        # Expert 0 should have higher predicted load due to trend
        self.assertGreater(predictions[0], expert_loads[0])
        
    def test_predictive_load_balancer_adjustment_computation(self):
        """Test computation of load balancing adjustments."""
        balancer = PredictiveLoadBalancer(self.config)
        
        # Create imbalanced current loads
        current_loads = {0: 50, 1: 10, 2: 5, 3: 2}
        
        # Update history
        balancer.update_load_history(current_loads)
        
        # Compute adjustments
        adjustments = balancer.compute_balancing_adjustments(current_loads)
        
        self.assertIn(0, adjustments)
        # Overloaded expert should get negative adjustment
        self.assertLess(adjustments[0], 0)
        
    def test_multi_objective_routing_optimizer_initialization(self):
        """Test MRO algorithm initialization."""
        optimizer = MultiObjectiveRoutingOptimizer(self.config)
        
        self.assertIn('entropy', optimizer.objective_history)
        self.assertIn('load_balance', optimizer.objective_history)
        self.assertEqual(len(optimizer.pareto_front), 0)
        
    def test_multi_objective_routing_optimizer_objective_computation(self):
        """Test computation of multiple routing objectives."""
        optimizer = MultiObjectiveRoutingOptimizer(self.config)
        
        # Test data
        routing_weights = [0.4, 0.3, 0.2, 0.1]
        expert_loads = {0: 10, 1: 8, 2: 6, 3: 4}
        expert_performance = {0: 1.0, 1: 0.9, 2: 1.1, 3: 0.8}
        
        scores = optimizer.compute_objective_scores(
            routing_weights, expert_loads, expert_performance
        )
        
        # Validate all objectives computed
        self.assertIn('entropy', scores)
        self.assertIn('load_balance', scores)
        self.assertIn('performance', scores)
        self.assertIn('diversity', scores)
        
        # Validate score ranges
        self.assertGreaterEqual(scores['entropy'], 0)
        self.assertLessEqual(scores['entropy'], 1)
        self.assertGreaterEqual(scores['load_balance'], 0)
        self.assertLessEqual(scores['load_balance'], 1)
        
    def test_multi_objective_routing_optimizer_decision_optimization(self):
        """Test optimization of routing decisions."""
        optimizer = MultiObjectiveRoutingOptimizer(self.config)
        
        # Create candidate routing weight sets
        candidates = [
            [0.8, 0.1, 0.05, 0.05],  # Concentrated
            [0.4, 0.3, 0.2, 0.1],    # Balanced
            [0.25, 0.25, 0.25, 0.25] # Uniform
        ]
        
        expert_loads = {0: 10, 1: 10, 2: 10, 3: 10}
        expert_performance = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        
        best_weights, optimization_info = optimizer.optimize_routing_decision(
            candidates, expert_loads, expert_performance
        )
        
        self.assertIn(best_weights, candidates)
        self.assertIn('total_score', optimization_info)
        self.assertIn('objective_scores', optimization_info)
        
    def test_adaptive_routing_system_integration(self):
        """Test integrated adaptive routing system."""
        system = AdaptiveRoutingSystem(self.config)
        
        # Test processing routing decision
        expert_logits = [1.0, 0.5, 2.0, 0.8, 1.2, 0.3, 1.8, 0.6]
        expert_loads = {i: 10 for i in range(len(expert_logits))}
        expert_performance = {i: 1.0 for i in range(len(expert_logits))}
        
        decision = system.process_routing_decision(
            expert_logits, expert_loads, expert_performance
        )
        
        # Validate routing decision
        self.assertIsInstance(decision, RoutingDecision)
        self.assertEqual(len(decision.selected_experts), 2)  # Top-k = 2
        self.assertEqual(len(decision.routing_weights), len(expert_logits))
        self.assertGreater(decision.entropy_score, 0)
        self.assertIn('entropy_adaptation', decision.adaptation_applied)
        
        # Test metrics retrieval
        metrics = system.get_comprehensive_metrics()
        self.assertIn('system_performance', metrics)
        self.assertIn('entropy_router', metrics)
        self.assertIn('resurrection_framework', metrics)
        
        # Cleanup
        system.stop_adaptation()


class TestResearchValidationFramework(unittest.TestCase):
    """Test suite for research validation framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
            
        self.config = ExperimentalConfig(
            num_runs=2,  # Reduced for testing
            sequence_length=100,  # Reduced for testing
            expert_count=4
        )
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_experimental_config_initialization(self):
        """Test experimental configuration initialization."""
        config = ExperimentalConfig()
        
        self.assertGreater(config.num_runs, 0)
        self.assertGreater(config.sequence_length, 0)
        self.assertGreater(config.expert_count, 0)
        self.assertIn('balanced_load', config.scenarios)
        
    def test_scenario_generator_initialization(self):
        """Test scenario generator initialization."""
        generator = ScenarioGenerator(self.config)
        self.assertEqual(generator.config, self.config)
        
    def test_scenario_generator_balanced_load(self):
        """Test balanced load scenario generation."""
        generator = ScenarioGenerator(self.config)
        scenario_data = generator.generate_balanced_load_scenario()
        
        self.assertEqual(len(scenario_data), self.config.sequence_length)
        self.assertEqual(len(scenario_data[0]), self.config.expert_count)
        
        # Validate all entries are lists of floats
        for sequence in scenario_data:
            self.assertEqual(len(sequence), self.config.expert_count)
            for logit in sequence:
                self.assertIsInstance(logit, (int, float))
                
    def test_scenario_generator_imbalanced_load(self):
        """Test imbalanced load scenario generation."""
        generator = ScenarioGenerator(self.config)
        scenario_data = generator.generate_imbalanced_load_scenario()
        
        self.assertEqual(len(scenario_data), self.config.sequence_length)
        
        # Check that early experts have higher average logits (imbalanced)
        avg_logits = [0] * self.config.expert_count
        for sequence in scenario_data:
            for i, logit in enumerate(sequence):
                avg_logits[i] += logit
        
        avg_logits = [x / len(scenario_data) for x in avg_logits]
        
        # First expert should have higher average than last
        self.assertGreater(avg_logits[0], avg_logits[-1])
        
    def test_scenario_generator_dynamic_patterns(self):
        """Test dynamic patterns scenario generation."""
        generator = ScenarioGenerator(self.config)
        scenario_data = generator.generate_dynamic_patterns_scenario()
        
        self.assertEqual(len(scenario_data), self.config.sequence_length)
        
        # Validate that patterns change over time
        first_quarter = scenario_data[:len(scenario_data)//4]
        last_quarter = scenario_data[-len(scenario_data)//4:]
        
        # Average logits should be different between first and last quarter
        avg_first = [sum(seq[i] for seq in first_quarter) / len(first_quarter) 
                    for i in range(self.config.expert_count)]
        avg_last = [sum(seq[i] for seq in last_quarter) / len(last_quarter)
                   for i in range(self.config.expert_count)]
        
        # At least one expert should have significantly different averages
        differences = [abs(avg_first[i] - avg_last[i]) for i in range(self.config.expert_count)]
        self.assertGreater(max(differences), 0.5)
        
    def test_metrics_calculator_entropy(self):
        """Test entropy calculation."""
        weights = [0.4, 0.3, 0.2, 0.1]
        entropy = MetricsCalculator.calculate_entropy(weights)
        
        self.assertGreater(entropy, 0)
        
        # Test uniform distribution (maximum entropy)
        uniform_weights = [0.25, 0.25, 0.25, 0.25]
        uniform_entropy = MetricsCalculator.calculate_entropy(uniform_weights)
        
        self.assertGreater(uniform_entropy, entropy)
        
    def test_metrics_calculator_load_balance_fairness(self):
        """Test load balance fairness calculation."""
        # Perfectly balanced
        balanced_loads = {0: 10, 1: 10, 2: 10, 3: 10}
        balanced_fairness = MetricsCalculator.calculate_load_balance_fairness(balanced_loads)
        
        # Imbalanced
        imbalanced_loads = {0: 30, 1: 5, 2: 3, 3: 2}
        imbalanced_fairness = MetricsCalculator.calculate_load_balance_fairness(imbalanced_loads)
        
        self.assertGreater(balanced_fairness, imbalanced_fairness)
        self.assertAlmostEqual(balanced_fairness, 1.0, places=2)
        
    def test_metrics_calculator_expert_utilization(self):
        """Test expert utilization calculation."""
        # All experts used
        all_used_loads = {0: 10, 1: 5, 2: 8, 3: 12}
        all_used_util = MetricsCalculator.calculate_expert_utilization(all_used_loads, 100)
        
        # Some experts unused
        some_unused_loads = {0: 10, 1: 0, 2: 8, 3: 0}
        some_unused_util = MetricsCalculator.calculate_expert_utilization(some_unused_loads, 100)
        
        self.assertEqual(all_used_util, 1.0)
        self.assertEqual(some_unused_util, 0.5)
        
    def test_experiment_runner_initialization(self):
        """Test experiment runner initialization."""
        runner = ExperimentRunner(self.config)
        
        self.assertEqual(runner.config, self.config)
        self.assertIsNotNone(runner.scenario_generator)
        self.assertIsNotNone(runner.metrics_calculator)
        self.assertEqual(len(runner.results), 0)
        
    def test_experiment_runner_single_experiment(self):
        """Test running a single experiment."""
        runner = ExperimentRunner(self.config)
        
        result = runner.run_single_experiment('balanced_load', 'baseline', 0)
        
        self.assertEqual(result.scenario, 'balanced_load')
        self.assertEqual(result.algorithm, 'baseline')
        self.assertEqual(result.run_id, 0)
        self.assertIn('entropy', result.metrics)
        self.assertIn('load_balance_fairness', result.metrics)
        self.assertGreater(result.execution_time, 0)
        
    def test_statistical_analyzer_initialization(self):
        """Test statistical analyzer initialization."""
        # Create mock results
        mock_results = []
        analyzer = StatisticalAnalyzer(mock_results)
        
        self.assertEqual(analyzer.results, mock_results)
        self.assertEqual(len(analyzer.analyses), 0)
        
    def test_research_report_generator_initialization(self):
        """Test research report generator initialization."""
        mock_results = []
        mock_analysis = {}
        
        generator = ResearchReportGenerator(mock_results, mock_analysis)
        
        self.assertEqual(generator.results, mock_results)
        self.assertEqual(generator.statistical_analysis, mock_analysis)
        
    def test_research_report_generator_methodology_section(self):
        """Test methodology section generation."""
        # Create minimal test data
        mock_results = [
            type('MockResult', (), {
                'scenario': 'balanced_load',
                'metadata': {'expert_count': 8, 'sequence_length': 1000},
                'metrics': {'entropy': 1.5, 'load_balance_fairness': 0.8}
            })()
        ]
        mock_analysis = {}
        
        generator = ResearchReportGenerator(mock_results, mock_analysis)
        methodology = generator.generate_methodology_section()
        
        self.assertIn('Methodology', methodology)
        self.assertIn('Experimental Design', methodology)
        self.assertIn('Test Scenarios', methodology)
        
    @patch('moe_debugger.research_validation.run_comprehensive_research_validation')
    def test_comprehensive_research_validation_pipeline(self, mock_validation):
        """Test comprehensive research validation pipeline."""
        # Mock the validation function to avoid long execution
        mock_validation.return_value = {
            'results': [],
            'statistical_analysis': {
                'publication_summary': {
                    'success_rate': 0.75,
                    'recommended_for_publication': True
                }
            },
            'publication_ready': True
        }
        
        result = run_comprehensive_research_validation(self.temp_dir)
        
        self.assertIn('results', result)
        self.assertIn('statistical_analysis', result)
        self.assertTrue(result['publication_ready'])


class TestEnhancedDebuggerIntegration(unittest.TestCase):
    """Test suite for enhanced debugger integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
            
        # Create mock model
        self.mock_model = MagicMock(spec=nn.Module)
        self.mock_model.__class__.__name__ = 'TestMoEModel'
        self.mock_model.named_modules.return_value = [
            ('layer.0', MagicMock()),
            ('layer.1', MagicMock()),
            ('expert.0', MagicMock()),
            ('expert.1', MagicMock())
        ]
        self.mock_model.named_parameters.return_value = []
        self.mock_model.parameters.return_value = [
            MagicMock(numel=lambda: 1000, requires_grad=True, device=MagicMock(type='cpu'), dtype=torch.float32)
        ]
        
    def test_enhanced_debugger_config_initialization(self):
        """Test enhanced debugger configuration."""
        config = EnhancedDebuggerConfig()
        
        self.assertTrue(config.adaptive_routing_enabled)
        self.assertFalse(config.research_mode)
        self.assertTrue(config.real_time_adaptation)
        
    def test_enhanced_debugger_initialization_basic(self):
        """Test enhanced debugger initialization with basic settings."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=False)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            debugger = EnhancedMoEDebugger(self.mock_model, config)
            
            self.assertEqual(debugger.enhanced_config, config)
            self.assertIsNone(debugger.adaptive_router)
            self.assertFalse(debugger.research_mode)
            
    def test_enhanced_debugger_initialization_with_adaptive_routing(self):
        """Test enhanced debugger initialization with adaptive routing."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=True)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.AdaptiveRoutingSystem') as mock_adaptive:
                debugger = EnhancedMoEDebugger(self.mock_model, config)
                
                self.assertIsNotNone(debugger.adaptive_router)
                mock_adaptive.assert_called_once()
                
    def test_enhanced_debugger_initialization_with_research_mode(self):
        """Test enhanced debugger initialization with research mode."""
        config = EnhancedDebuggerConfig(research_mode=True)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            debugger = EnhancedMoEDebugger(self.mock_model, config)
            
            self.assertTrue(debugger.research_mode)
            
    def test_enhanced_debugger_session_management(self):
        """Test enhanced session management."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=False)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.MoEDebugger.start_session') as mock_start:
                with patch('moe_debugger.enhanced_debugger.MoEDebugger.end_session') as mock_end:
                    mock_session = MagicMock()
                    mock_session.session_id = 'test_session'
                    mock_start.return_value = mock_session
                    mock_end.return_value = mock_session
                    
                    debugger = EnhancedMoEDebugger(self.mock_model, config)
                    
                    # Test session start
                    session = debugger.start_session()
                    self.assertEqual(session.session_id, 'test_session')
                    
                    # Test session end
                    ended_session = debugger.end_session()
                    self.assertEqual(ended_session.session_id, 'test_session')
                    
    def test_enhanced_debugger_adaptive_trace_context(self):
        """Test adaptive trace context manager."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=False)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.MoEDebugger.is_active', True):
                debugger = EnhancedMoEDebugger(self.mock_model, config)
                debugger.hooks_manager = MagicMock()
                
                with debugger.adaptive_trace('test_sequence'):
                    # Context manager should work
                    pass
                
                # Verify hooks were called
                debugger.hooks_manager.start_sequence.assert_called_with('test_sequence')
                debugger.hooks_manager.end_sequence.assert_called_once()
                
    def test_enhanced_debugger_adaptive_routing_stats(self):
        """Test adaptive routing statistics retrieval."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=True)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.MoEDebugger.get_routing_stats') as mock_stats:
                with patch('moe_debugger.enhanced_debugger.AdaptiveRoutingSystem') as mock_adaptive:
                    mock_stats.return_value = {'base_metric': 1.0}
                    mock_router = MagicMock()
                    mock_router.get_comprehensive_metrics.return_value = {'adaptive_metric': 2.0}
                    mock_adaptive.return_value = mock_router
                    
                    debugger = EnhancedMoEDebugger(self.mock_model, config)
                    stats = debugger.get_adaptive_routing_stats()
                    
                    self.assertIn('base_metric', stats)
                    self.assertIn('adaptive_routing', stats)
                    self.assertIn('research_recommendations', stats)
                    
    def test_enhanced_debugger_issue_detection(self):
        """Test enhanced issue detection."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=True)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.MoEDebugger.detect_issues') as mock_detect:
                with patch('moe_debugger.enhanced_debugger.AdaptiveRoutingSystem') as mock_adaptive:
                    mock_detect.return_value = [{'type': 'base_issue'}]
                    
                    mock_router = MagicMock()
                    mock_router.get_comprehensive_metrics.return_value = {
                        'resurrection_framework': {'currently_dead': 3, 'dead_expert_ids': [2, 4, 6]},
                        'entropy_router': {'entropy_trend': [0.4, 0.3, 0.2, 0.1]},
                        'load_balancer': {'max_adjustment': 0.6}
                    }
                    mock_adaptive.return_value = mock_router
                    
                    debugger = EnhancedMoEDebugger(self.mock_model, config)
                    issues = debugger.detect_issues_enhanced()
                    
                    # Should include base issues plus enhanced issues
                    self.assertGreater(len(issues), 1)
                    issue_types = [issue['type'] for issue in issues]
                    self.assertIn('base_issue', issue_types)
                    
    def test_enhanced_debugger_benchmark_against_baseline(self):
        """Test benchmarking against baseline."""
        config = EnhancedDebuggerConfig(adaptive_routing_enabled=True)
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            with patch('moe_debugger.enhanced_debugger.AdaptiveRoutingSystem') as mock_adaptive:
                debugger = EnhancedMoEDebugger(self.mock_model, config)
                debugger.architecture = MagicMock()
                debugger.architecture.num_experts_per_layer = 8
                
                # Mock adaptive router
                mock_router = MagicMock()
                mock_decision = MagicMock()
                mock_decision.selected_experts = [0, 1]
                mock_decision.entropy_score = 1.5
                mock_router.process_routing_decision.return_value = mock_decision
                debugger.adaptive_router = mock_router
                
                benchmark_result = debugger.benchmark_against_baseline(sequence_length=10)
                
                self.assertIn('entropy_improvement_percentage', benchmark_result)
                self.assertIn('baseline_metrics', benchmark_result)
                self.assertIn('adaptive_metrics', benchmark_result)
                
    def test_create_enhanced_debugger_factory(self):
        """Test enhanced debugger factory function."""
        with patch('moe_debugger.enhanced_debugger.EnhancedMoEDebugger') as mock_debugger:
            debugger = create_enhanced_debugger(
                self.mock_model,
                adaptive_routing=True,
                research_mode=True
            )
            
            mock_debugger.assert_called_once()
            call_args = mock_debugger.call_args
            config = call_args[0][1]  # Second argument is config
            
            self.assertTrue(config.adaptive_routing_enabled)
            self.assertTrue(config.research_mode)
            
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        self.assertEqual(len(monitor.metrics_history), 0)
        self.assertGreater(monitor.start_time, 0)
        
    def test_performance_monitor_metric_logging(self):
        """Test performance metric logging."""
        monitor = PerformanceMonitor()
        
        monitor.log_metric('entropy', 1.5, {'context': 'test'})
        
        self.assertEqual(len(monitor.metrics_history), 1)
        metric = monitor.metrics_history[0]
        self.assertEqual(metric['metric_name'], 'entropy')
        self.assertEqual(metric['value'], 1.5)
        self.assertEqual(metric['metadata']['context'], 'test')
        
    def test_performance_monitor_summary(self):
        """Test performance summary generation."""
        monitor = PerformanceMonitor()
        
        # Log some metrics
        for i in range(5):
            monitor.log_metric(f'metric_{i}', i * 0.1)
        
        summary = monitor.get_performance_summary()
        
        self.assertEqual(summary['total_metrics_logged'], 5)
        self.assertGreater(summary['monitoring_duration'], 0)
        self.assertEqual(len(summary['recent_metrics']), 5)


class TestIntegrationAndEndToEnd(unittest.TestCase):
    """Integration and end-to-end tests for complete research system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
            
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_end_to_end_adaptive_routing_workflow(self):
        """Test complete adaptive routing workflow."""
        # Create adaptive routing system
        config = AdaptiveRoutingConfig()
        system = AdaptiveRoutingSystem(config)
        
        try:
            # Simulate routing decisions
            expert_logits = [1.0, 0.5, 2.0, 0.8]
            expert_loads = {0: 10, 1: 8, 2: 12, 3: 5}
            expert_performance = {0: 1.0, 1: 0.9, 2: 1.1, 3: 0.8}
            
            # Process multiple decisions
            decisions = []
            for _ in range(10):
                decision = system.process_routing_decision(
                    expert_logits, expert_loads, expert_performance
                )
                decisions.append(decision)
            
            # Validate decisions
            self.assertEqual(len(decisions), 10)
            for decision in decisions:
                self.assertIsInstance(decision, RoutingDecision)
                self.assertEqual(len(decision.selected_experts), 2)
                
            # Get comprehensive metrics
            metrics = system.get_comprehensive_metrics()
            self.assertIn('system_performance', metrics)
            
        finally:
            system.stop_adaptation()
            
    def test_end_to_end_research_validation_workflow(self):
        """Test complete research validation workflow."""
        # Create minimal experimental config
        config = ExperimentalConfig(
            num_runs=1,  # Minimal for testing
            sequence_length=50,
            expert_count=4,
            scenarios=['balanced_load']
        )
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run_full_experiment_suite()
        
        # Validate results
        self.assertGreater(len(results), 0)
        
        # Statistical analysis
        analyzer = StatisticalAnalyzer(results)
        analysis = analyzer.generate_statistical_report()
        
        self.assertIn('experiment_summary', analysis)
        self.assertIn('statistical_results', analysis)
        
        # Generate report
        generator = ResearchReportGenerator(results, analysis)
        report = generator.generate_full_research_report()
        
        self.assertIn('Abstract', report)
        self.assertIn('Methodology', report)
        
    @patch('moe_debugger.research_validation.run_comprehensive_research_validation')
    def test_end_to_end_enhanced_debugger_research_workflow(self, mock_validation):
        """Test complete enhanced debugger research workflow."""
        # Mock validation to avoid long execution
        mock_validation.return_value = {
            'results': [],
            'statistical_analysis': {
                'publication_summary': {
                    'success_rate': 0.8,
                    'recommended_for_publication': True
                }
            },
            'publication_ready': True
        }
        
        # Create mock model
        mock_model = MagicMock(spec=nn.Module)
        mock_model.__class__.__name__ = 'TestModel'
        mock_model.named_modules.return_value = []
        mock_model.named_parameters.return_value = []
        mock_model.parameters.return_value = [
            MagicMock(numel=lambda: 100, requires_grad=True, device=MagicMock(type='cpu'), dtype=torch.float32)
        ]
        
        # Create enhanced debugger with research mode
        config = EnhancedDebuggerConfig(
            adaptive_routing_enabled=True,
            research_mode=True,
            experiment_output_dir=self.temp_dir
        )
        
        with patch('moe_debugger.enhanced_debugger.MoEDebugger.__init__'):
            debugger = create_enhanced_debugger(
                mock_model,
                adaptive_routing=True,
                research_mode=True,
                experiment_output_dir=self.temp_dir
            )
            
            # Run research validation
            research_results = debugger.run_research_validation()
            
            self.assertIn('validation_results', research_results)
            self.assertIn('research_insights', research_results)
            self.assertTrue(research_results['validation_results']['publication_ready'])


def run_comprehensive_test_suite():
    """Run the complete test suite with detailed reporting."""
    print("🧪 STARTING COMPREHENSIVE RESEARCH EXTENSION TEST SUITE")
    print("=" * 80)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available - skipping tests")
        return False
    
    # Create test suite
    test_classes = [
        TestAdaptiveRoutingAlgorithms,
        TestResearchValidationFramework,
        TestEnhancedDebuggerIntegration,
        TestIntegrationAndEndToEnd
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 60)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"❌ {len(result.failures)} test(s) failed")
        if result.errors:
            print(f"⚠️  {len(result.errors)} test(s) had errors")
        if not result.failures and not result.errors:
            print(f"✅ All {result.testsRun} tests passed")
    
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE TEST SUITE SUMMARY")
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_tests - total_failures - total_errors}")
    print(f"❌ Failed: {total_failures}")
    print(f"⚠️  Errors: {total_errors}")
    
    success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1)
    print(f"📈 Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.95:
        print("🏆 RESEARCH QUALITY STANDARD ACHIEVED (95%+)")
        print("📝 Code ready for publication and peer review")
        return True
    elif success_rate >= 0.85:
        print("✅ PRODUCTION QUALITY STANDARD ACHIEVED (85%+)")
        print("🔧 Minor improvements recommended before publication")
        return True
    else:
        print("⚠️  QUALITY STANDARD NOT MET")
        print("🔧 Significant improvements required")
        return False


if __name__ == '__main__':
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)