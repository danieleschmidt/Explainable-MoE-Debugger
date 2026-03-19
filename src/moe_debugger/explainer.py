"""RoutingExplainer: gradient-based attribution for MoE routing decisions.

Uses vanilla gradients (∂gate_logit / ∂input) to identify which input
features most influenced each token's routing choice.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .layer import MoELayer


class RoutingExplainer:
    """Explains why tokens were routed to specific experts.

    Strategy: compute gradient of the gate logit for each expert
    w.r.t. the token input. High-magnitude gradient dimensions are
    the features the router "looks at" most when deciding to send
    a token to that expert.

    Usage::

        layer = MoELayer(d_model=64, d_ffn=128, num_experts=4, top_k=2)
        explainer = RoutingExplainer(layer)

        # Run a batch through the layer
        x = torch.randn(8, 16, 64)
        out = layer(x)

        # Get per-expert attribution for the tokens it received
        attrs = explainer.compute_attribution()
        # attrs[expert_id] → Tensor of shape (n_routed_tokens, d_model)
    """

    def __init__(self, layer: MoELayer) -> None:
        self.layer = layer

    def compute_attribution(
        self,
        normalize: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """Compute gradient-based attribution for the last recorded forward pass.

        For each expert, we compute:
            attr[i, j] = |∂ gate_logit_expert / ∂ token_input[i, j]|

        for every token routed to that expert.

        Args:
            normalize: If True, L1-normalize attributions per token.

        Returns:
            Dict mapping expert_id → attribution tensor (n_routed, d_model).
            Experts that received no tokens are absent from the dict.
        """
        record = self.layer.last_record
        if record is None:
            raise RuntimeError(
                "No routing record found. Run a forward pass through the MoELayer first."
            )

        attributions: Dict[int, torch.Tensor] = {}

        for expert_id in range(self.layer.num_experts):
            # Which tokens were routed to this expert?
            mask = record.gate_weights[:, expert_id] > 0
            if not mask.any():
                continue

            # Recompute gate logit with grad enabled (can't diff through detach'd tensors)
            token_inputs = record.token_inputs[mask].clone().requires_grad_(True)
            logit = self.layer.gate(token_inputs)[:, expert_id]  # (n_routed,)

            # Sum and backprop — gives ∂sum(logit) / ∂input
            logit.sum().backward()
            grad = token_inputs.grad  # (n_routed, d_model)

            attr = grad.abs()
            if normalize:
                norm = attr.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                attr = attr / norm

            attributions[expert_id] = attr.detach()

        return attributions

    def top_features_per_expert(
        self,
        top_k: int = 10,
        feature_names: list[str] | None = None,
    ) -> Dict[int, list]:
        """Return the top-k most important feature indices (or names) per expert.

        Args:
            top_k:         Number of top features to return per expert.
            feature_names: Optional list of names for each feature dimension.

        Returns:
            Dict mapping expert_id → list of (feature_index_or_name, mean_attribution).
        """
        attributions = self.compute_attribution(normalize=True)
        result: Dict[int, list] = {}

        for expert_id, attr in attributions.items():
            mean_attr = attr.mean(dim=0)  # (d_model,)
            top_vals, top_idx = torch.topk(mean_attr, min(top_k, mean_attr.shape[0]))
            features = []
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                name = feature_names[idx] if feature_names else idx
                features.append((name, round(val, 6)))
            result[expert_id] = features

        return result
