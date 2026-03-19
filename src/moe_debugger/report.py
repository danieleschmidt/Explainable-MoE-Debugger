"""MoEDebugReport: structured diagnostic report for a MoE layer.

Aggregates all debugger components into a single, human-readable report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .activation_map import ExpertActivationMap
from .dead_expert import DeadExpertDetector, ExpertUtilizationStats
from .explainer import RoutingExplainer
from .layer import MoELayer


@dataclass
class ExpertReport:
    """Summary for a single expert."""
    expert_id: int
    utilization_rate: float
    is_dead: bool
    is_overloaded: bool
    top_routing_features: List  # from RoutingExplainer
    top_activation_features: List  # from ExpertActivationMap


@dataclass
class MoEDebugReport:
    """Full diagnostic report for a MoE layer.

    Attributes:
        layer_info:      Configuration of the MoE layer.
        expert_reports:  Per-expert diagnostics.
        dead_experts:    IDs of dead experts.
        overloaded_experts: IDs of overloaded experts.
        routing_entropy: Mean routing entropy (higher = more balanced).
        load_imbalance:  Coefficient of variation across expert utilization rates.
        raw_utilization: Dict of expert_id → utilization rate.
    """

    layer_info: Dict[str, Any]
    expert_reports: List[ExpertReport]
    dead_experts: List[int]
    overloaded_experts: List[int]
    routing_entropy: float
    load_imbalance: float
    raw_utilization: Dict[int, float]

    @classmethod
    def build(
        cls,
        layer: MoELayer,
        dead_expert_detector: Optional[DeadExpertDetector] = None,
        routing_explainer: Optional[RoutingExplainer] = None,
        activation_map: Optional[ExpertActivationMap] = None,
        top_k_features: int = 5,
        feature_names: Optional[List[str]] = None,
    ) -> "MoEDebugReport":
        """Build a report from a layer and optional debugger components.

        At minimum you need a layer that has run at least one forward pass.

        Args:
            layer:               The MoELayer to report on.
            dead_expert_detector: If provided, use its accumulated stats.
            routing_explainer:   If provided, include routing attribution.
            activation_map:      If provided, include activation profiles.
            top_k_features:      How many top features to include per expert.
            feature_names:       Optional names for input features.

        Returns:
            A fully populated MoEDebugReport.
        """
        # --- Layer info ---
        layer_info = {
            "d_model": layer.d_model,
            "d_ffn": layer.d_ffn,
            "num_experts": layer.num_experts,
            "top_k": layer.top_k,
        }

        # --- Utilization stats ---
        if dead_expert_detector is not None:
            util_stats: List[ExpertUtilizationStats] = dead_expert_detector.get_stats()
        else:
            # Build minimal stats from the last routing record
            record = layer.last_record
            if record is None:
                raise RuntimeError("Layer has no routing record. Run a forward pass first.")
            util_stats = _minimal_util_stats(layer, record)

        dead_experts = [s.expert_id for s in util_stats if s.is_dead]
        overloaded_experts = [s.expert_id for s in util_stats if s.is_overloaded]
        raw_util = {s.expert_id: round(s.utilization_rate, 6) for s in util_stats}

        # --- Routing entropy and load imbalance ---
        record = layer.last_record
        routing_entropy = _compute_routing_entropy(record.gate_weights)
        rates = torch.tensor(list(raw_util.values()), dtype=torch.float32)
        mean_rate = rates.mean().item()
        std_rate = rates.std().item()
        load_imbalance = std_rate / max(mean_rate, 1e-8)

        # --- Attribution ---
        routing_feats: Dict[int, list] = {}
        if routing_explainer is not None:
            try:
                routing_feats = routing_explainer.top_features_per_expert(
                    top_k=top_k_features, feature_names=feature_names
                )
            except Exception:
                pass  # silently skip if no grad available

        activation_feats: Dict[int, list] = {}
        if activation_map is not None:
            activation_feats = activation_map.top_activating_features(
                top_k=top_k_features, feature_names=feature_names
            )

        # --- Per-expert reports ---
        expert_reports = []
        for s in util_stats:
            expert_reports.append(ExpertReport(
                expert_id=s.expert_id,
                utilization_rate=round(s.utilization_rate, 6),
                is_dead=s.is_dead,
                is_overloaded=s.is_overloaded,
                top_routing_features=routing_feats.get(s.expert_id, []),
                top_activation_features=activation_feats.get(s.expert_id, []),
            ))

        return cls(
            layer_info=layer_info,
            expert_reports=expert_reports,
            dead_experts=dead_experts,
            overloaded_experts=overloaded_experts,
            routing_entropy=round(routing_entropy, 4),
            load_imbalance=round(load_imbalance, 4),
            raw_utilization=raw_util,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to a plain dict (JSON-friendly)."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print a concise human-readable summary to stdout."""
        cfg = self.layer_info
        print("=" * 60)
        print("  MoE Debug Report")
        print("=" * 60)
        print(
            f"  Layer: d_model={cfg['d_model']}, d_ffn={cfg['d_ffn']}, "
            f"num_experts={cfg['num_experts']}, top_k={cfg['top_k']}"
        )
        print(f"  Routing entropy:  {self.routing_entropy:.4f}")
        print(f"  Load imbalance (CV): {self.load_imbalance:.4f}  "
              f"(0=perfect balance, >0.5=concerning)")
        print()

        if self.dead_experts:
            print(f"  ⚠ Dead experts:      {self.dead_experts}")
        else:
            print("  ✓ No dead experts")

        if self.overloaded_experts:
            print(f"  ⚠ Overloaded experts: {self.overloaded_experts}")
        else:
            print("  ✓ No overloaded experts")

        print()
        print("  Expert Utilization:")
        for exp in self.expert_reports:
            status = ""
            if exp.is_dead:
                status = " [DEAD]"
            elif exp.is_overloaded:
                status = " [OVERLOADED]"
            bar_len = int(exp.utilization_rate * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(
                f"    Expert {exp.expert_id:2d}: {bar} "
                f"{exp.utilization_rate*100:5.1f}%{status}"
            )

        print()
        print("  Top Routing Features (what the router looks at per expert):")
        for exp in self.expert_reports:
            if exp.top_routing_features:
                feat_str = ", ".join(
                    f"{f}({v:.3f})" for f, v in exp.top_routing_features[:3]
                )
                print(f"    Expert {exp.expert_id:2d}: {feat_str}")

        print()
        print("  Top Activation Features (what each expert has learned to process):")
        for exp in self.expert_reports:
            if exp.top_activation_features:
                feat_str = ", ".join(
                    f"{f}({v:.3f})" for f, v in exp.top_activation_features[:3]
                )
                print(f"    Expert {exp.expert_id:2d}: {feat_str}")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_routing_entropy(gate_weights: torch.Tensor) -> float:
    """Compute mean entropy of gate weight distributions over tokens."""
    # gate_weights: (N, num_experts)
    # Only consider the non-zero portion
    eps = 1e-9
    gw = gate_weights.float()
    # Entropy per token
    entropy = -(gw * (gw + eps).log()).sum(dim=-1)  # (N,)
    return entropy.mean().item()


def _minimal_util_stats(
    layer: MoELayer, record: Any
) -> List[ExpertUtilizationStats]:
    """Build minimal utilization stats from a single routing record."""
    from .dead_expert import ExpertUtilizationStats

    n_tokens = record.gate_weights.shape[0]
    ideal_rate = layer.top_k / layer.num_experts

    stats = []
    for expert_id in range(layer.num_experts):
        n_routed = int((record.gate_weights[:, expert_id] > 0).sum().item())
        rate = n_routed / max(n_tokens, 1)
        stats.append(ExpertUtilizationStats(
            expert_id=expert_id,
            total_tokens_seen=n_tokens,
            total_tokens_routed=n_routed,
            utilization_rate=rate,
            batch_utilization_history=[rate],
            is_dead=False,
            is_overloaded=False,
        ))
    return stats
