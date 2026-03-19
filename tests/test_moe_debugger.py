"""Tests for the Explainable-MoE-Debugger core components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_debugger import (
    MoELayer,
    RoutingExplainer,
    ExpertActivationMap,
    DeadExpertDetector,
    MoEDebugReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layer():
    torch.manual_seed(0)
    return MoELayer(d_model=32, d_ffn=64, num_experts=4, top_k=2, noise_std=0.0)


@pytest.fixture
def layer_with_data(layer):
    x = torch.randn(8, 16, 32)
    with torch.no_grad():
        out = layer(x)
    return layer, x, out


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

class TestMoELayer:
    def test_output_shape_3d(self, layer):
        x = torch.randn(4, 10, 32)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == x.shape

    def test_output_shape_2d(self, layer):
        x = torch.randn(16, 32)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == x.shape

    def test_routing_record_populated(self, layer):
        x = torch.randn(6, 8, 32)
        with torch.no_grad():
            layer(x)
        rec = layer.last_record
        assert rec is not None
        assert rec.gate_weights.shape == (6 * 8, 4)
        assert rec.expert_indices.shape == (6 * 8, 2)
        assert rec.token_inputs.shape == (6 * 8, 32)

    def test_gate_weights_sum_to_one(self, layer):
        x = torch.randn(16, 32)
        with torch.no_grad():
            layer(x)
        weights = layer.last_record.gate_weights
        # Each row sums to top_k weights (top_k=2), which after softmax sum to 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_top_k_experts_selected(self, layer):
        x = torch.randn(16, 32)
        with torch.no_grad():
            layer(x)
        # Exactly top_k experts per token should have nonzero weight
        nonzero_per_token = (layer.last_record.gate_weights > 0).sum(dim=-1)
        assert (nonzero_per_token == 2).all()

    def test_no_record_when_disabled(self):
        layer = MoELayer(d_model=16, d_ffn=32, num_experts=2, top_k=1, record_routing=False)
        with torch.no_grad():
            layer(torch.randn(4, 16))
        assert layer.last_record is None

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            MoELayer(d_model=16, d_ffn=32, num_experts=2, top_k=5)

    def test_gradient_flows(self, layer):
        x = torch.randn(4, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()


# ---------------------------------------------------------------------------
# RoutingExplainer
# ---------------------------------------------------------------------------

class TestRoutingExplainer:
    def test_returns_dict_of_experts(self, layer_with_data):
        layer, x, _ = layer_with_data
        explainer = RoutingExplainer(layer)
        attrs = explainer.compute_attribution()
        # Should have entries for experts that received tokens
        assert len(attrs) > 0
        assert all(isinstance(k, int) for k in attrs)

    def test_attribution_shape(self, layer_with_data):
        layer, x, _ = layer_with_data
        explainer = RoutingExplainer(layer)
        attrs = explainer.compute_attribution()
        for expert_id, attr in attrs.items():
            # Should be (n_routed_tokens, d_model)
            assert attr.dim() == 2
            assert attr.shape[-1] == layer.d_model

    def test_normalized_attribution_sums_to_one(self, layer_with_data):
        layer, x, _ = layer_with_data
        explainer = RoutingExplainer(layer)
        attrs = explainer.compute_attribution(normalize=True)
        for attr in attrs.values():
            row_sums = attr.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_top_features_returns_list(self, layer_with_data):
        layer, x, _ = layer_with_data
        explainer = RoutingExplainer(layer)
        top = explainer.top_features_per_expert(top_k=5)
        for expert_id, feats in top.items():
            assert isinstance(feats, list)
            assert len(feats) <= 5
            for name, val in feats:
                assert isinstance(val, float)

    def test_error_without_record(self):
        layer = MoELayer(d_model=16, d_ffn=32, num_experts=2, top_k=1)
        explainer = RoutingExplainer(layer)
        with pytest.raises(RuntimeError):
            explainer.compute_attribution()

    def test_feature_names_used(self, layer_with_data):
        layer, x, _ = layer_with_data
        names = [f"feat_{i}" for i in range(layer.d_model)]
        explainer = RoutingExplainer(layer)
        top = explainer.top_features_per_expert(top_k=3, feature_names=names)
        for feats in top.values():
            for name, _ in feats:
                assert name.startswith("feat_")


# ---------------------------------------------------------------------------
# ExpertActivationMap
# ---------------------------------------------------------------------------

class TestExpertActivationMap:
    def test_profiles_shape(self, layer_with_data):
        layer, x, _ = layer_with_data
        act_map = ExpertActivationMap(layer)
        act_map.update()
        profiles = act_map.get_profiles()
        for expert_id, profile in profiles.items():
            assert profile.shape == (layer.d_model,)

    def test_normalized_profiles_sum_to_one(self, layer_with_data):
        layer, x, _ = layer_with_data
        act_map = ExpertActivationMap(layer)
        act_map.update()
        profiles = act_map.get_profiles(normalize=True)
        for profile in profiles.values():
            assert abs(profile.sum().item() - 1.0) < 1e-5

    def test_multiple_batches_accumulated(self, layer):
        act_map = ExpertActivationMap(layer)
        for _ in range(5):
            with torch.no_grad():
                layer(torch.randn(8, 32))
            act_map.update()
        profiles = act_map.get_profiles()
        # After 5 batches, should still have valid profiles
        assert len(profiles) > 0

    def test_reset_clears_state(self, layer_with_data):
        layer, x, _ = layer_with_data
        act_map = ExpertActivationMap(layer)
        act_map.update()
        act_map.reset()
        profiles = act_map.get_profiles()
        assert len(profiles) == 0

    def test_top_activating_features(self, layer_with_data):
        layer, x, _ = layer_with_data
        act_map = ExpertActivationMap(layer)
        act_map.update()
        top = act_map.top_activating_features(top_k=5)
        for expert_id, feats in top.items():
            assert len(feats) <= 5


# ---------------------------------------------------------------------------
# DeadExpertDetector
# ---------------------------------------------------------------------------

class TestDeadExpertDetector:
    def test_stats_have_all_experts(self, layer_with_data):
        layer, x, _ = layer_with_data
        detector = DeadExpertDetector(layer)
        detector.update()
        stats = detector.get_stats()
        assert len(stats) == layer.num_experts

    def test_utilization_sums_to_top_k_over_n(self, layer):
        """Total utilization rates should sum to top_k / num_experts * num_experts = top_k."""
        with torch.no_grad():
            layer(torch.randn(1000, 32))
        detector = DeadExpertDetector(layer)
        detector.update()
        stats = detector.get_stats()
        total = sum(s.utilization_rate for s in stats)
        # Sum of rates should ≈ top_k (each token contributes top_k expert selections)
        expected = layer.top_k
        assert abs(total - expected) < 0.1, f"Expected ~{expected}, got {total:.4f}"

    def test_dead_expert_detection(self):
        """Force a dead expert by manually injecting utilization history."""
        layer = MoELayer(d_model=16, d_ffn=32, num_experts=4, top_k=2, noise_std=0.0)
        detector = DeadExpertDetector(
            layer,
            utilization_threshold=0.05,
            window_size=10,
        )
        # Manually populate the window: experts 0,1,3 healthy; expert 2 near-zero
        for _ in range(10):
            detector._windows[0].append(0.4)
            detector._windows[1].append(0.35)
            detector._windows[2].append(0.001)   # dead
            detector._windows[3].append(0.25)
            detector._total_tokens += 100
            detector._expert_tokens[0] += 40
            detector._expert_tokens[1] += 35
            detector._expert_tokens[2] += 0
            detector._expert_tokens[3] += 25

        dead = detector.dead_experts()
        assert 2 in dead, f"Expert 2 should be dead, got dead={dead}"
        assert 0 not in dead
        assert 1 not in dead
        assert 3 not in dead

    def test_reset(self, layer_with_data):
        layer, x, _ = layer_with_data
        detector = DeadExpertDetector(layer)
        detector.update()
        detector.reset()
        stats = detector.get_stats()
        for s in stats:
            assert s.utilization_rate == 0.0


# ---------------------------------------------------------------------------
# MoEDebugReport
# ---------------------------------------------------------------------------

class TestMoEDebugReport:
    def test_build_minimal(self, layer_with_data):
        layer, x, _ = layer_with_data
        report = MoEDebugReport.build(layer=layer)
        assert report.layer_info["num_experts"] == layer.num_experts
        assert len(report.expert_reports) == layer.num_experts

    def test_build_full(self, layer):
        # Collect a few batches
        detector = DeadExpertDetector(layer, window_size=10)
        act_map = ExpertActivationMap(layer)
        for _ in range(5):
            with torch.no_grad():
                layer(torch.randn(16, 32))
            detector.update()
            act_map.update()

        explainer = RoutingExplainer(layer)
        report = MoEDebugReport.build(
            layer=layer,
            dead_expert_detector=detector,
            routing_explainer=explainer,
            activation_map=act_map,
            top_k_features=3,
        )
        assert 0.0 <= report.routing_entropy
        assert report.load_imbalance >= 0.0

    def test_to_json_roundtrip(self, layer_with_data):
        layer, x, _ = layer_with_data
        report = MoEDebugReport.build(layer=layer)
        import json
        j = report.to_json()
        d = json.loads(j)
        assert "layer_info" in d
        assert "expert_reports" in d
        assert "dead_experts" in d

    def test_print_summary_runs(self, layer_with_data, capsys):
        layer, x, _ = layer_with_data
        report = MoEDebugReport.build(layer=layer)
        report.print_summary()
        captured = capsys.readouterr()
        assert "MoE Debug Report" in captured.out
        assert "Routing entropy" in captured.out

    def test_error_without_record(self):
        layer = MoELayer(d_model=16, d_ffn=32, num_experts=2, top_k=1)
        with pytest.raises(RuntimeError):
            MoEDebugReport.build(layer=layer)
