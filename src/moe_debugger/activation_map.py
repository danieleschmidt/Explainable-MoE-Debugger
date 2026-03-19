"""ExpertActivationMap: which input features activate each expert most strongly.

Different from the routing explainer (which explains the router's decision),
this shows what patterns within the expert's FFN respond most strongly —
i.e., what the expert has *learned* to process.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .layer import MoELayer


class ExpertActivationMap:
    """Profiles which input features activate each expert's hidden layer most.

    Strategy: for each expert, compute the mean absolute activation of the
    first hidden layer across all tokens routed to it. Then attribute those
    activations back to input dimensions via the weight matrix.

    Usage::

        layer = MoELayer(d_model=64, d_ffn=128, num_experts=4, top_k=2)
        act_map = ExpertActivationMap(layer)

        # Accumulate statistics over multiple batches
        for x, _ in dataloader:
            layer(x)
            act_map.update()

        # Get activation profiles
        profiles = act_map.get_profiles()
        # profiles[expert_id] → tensor of shape (d_model,) - input feature importance
    """

    def __init__(self, layer: MoELayer) -> None:
        self.layer = layer
        self._reset()

    def _reset(self) -> None:
        self._accumulated: Dict[int, torch.Tensor] = {}   # expert_id → sum of attr
        self._counts: Dict[int, int] = {}                  # expert_id → n_tokens

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._reset()

    def update(self) -> None:
        """Process the most recent forward pass recorded in the MoELayer.

        Call this after each forward pass to accumulate statistics.
        """
        record = self.layer.last_record
        if record is None:
            raise RuntimeError(
                "No routing record found. Run a forward pass through the MoELayer first."
            )

        for expert_id, expert in enumerate(self.layer.experts):
            mask = record.gate_weights[:, expert_id] > 0
            if not mask.any():
                continue

            token_inputs = record.token_inputs[mask]  # (n_routed, d_model)

            # First linear layer weights: (d_ffn, d_model)
            w = expert[0].weight.detach()  # (d_ffn, d_model)

            # Hidden activations (pre-GELU): (n_routed, d_ffn)
            hidden = token_inputs @ w.T

            # Attribute hidden activation back to input features:
            # For each hidden unit h, it activates with value w_h · x.
            # Importance of input feature j = sum_h |w_hj| * mean(|hidden_h|)
            mean_hidden_abs = hidden.abs().mean(dim=0)  # (d_ffn,)
            # (d_model,) = sum over d_ffn of weight contribution
            input_importance = (w.abs() * mean_hidden_abs.unsqueeze(-1)).sum(dim=0)

            if expert_id not in self._accumulated:
                self._accumulated[expert_id] = torch.zeros(
                    self.layer.d_model, dtype=token_inputs.dtype
                )
                self._counts[expert_id] = 0

            self._accumulated[expert_id] += input_importance
            self._counts[expert_id] += 1

    def get_profiles(self, normalize: bool = True) -> Dict[int, torch.Tensor]:
        """Return mean activation profile per expert.

        Args:
            normalize: L1-normalize so profiles sum to 1.

        Returns:
            Dict mapping expert_id → input importance vector of shape (d_model,).
        """
        profiles: Dict[int, torch.Tensor] = {}
        for expert_id, acc in self._accumulated.items():
            profile = acc / max(self._counts[expert_id], 1)
            if normalize:
                norm = profile.sum().clamp(min=1e-8)
                profile = profile / norm
            profiles[expert_id] = profile
        return profiles

    def top_activating_features(
        self,
        top_k: int = 10,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[int, List[Tuple]]:
        """Return the top-k most activating input features per expert.

        Args:
            top_k:         Number of features to return per expert.
            feature_names: Optional list of names for each feature dimension.

        Returns:
            Dict mapping expert_id → list of (feature_index_or_name, mean_importance).
        """
        profiles = self.get_profiles(normalize=True)
        result: Dict[int, List[Tuple]] = {}
        for expert_id, profile in profiles.items():
            top_vals, top_idx = torch.topk(profile, min(top_k, profile.shape[0]))
            features = []
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                name = feature_names[idx] if feature_names else idx
                features.append((name, round(val, 6)))
            result[expert_id] = features
        return result
