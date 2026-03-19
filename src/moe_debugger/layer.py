"""MoELayer: Mixture-of-Experts layer with built-in routing recorder.

Records per-token routing decisions (which expert, gate weights) for
downstream explainability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingRecord:
    """A snapshot of routing decisions for one forward pass.

    Attributes:
        gate_logits:  Raw logits from the router, shape (batch*seq, num_experts).
        gate_weights: Softmax'd weights AFTER top-k masking, shape (batch*seq, num_experts).
        expert_indices: Which top-k experts were selected per token, shape (batch*seq, top_k).
        token_inputs:   The input tensor that was routed, shape (batch*seq, d_model).
                        Retained so explainability tools can compute gradients.
    """
    gate_logits: torch.Tensor
    gate_weights: torch.Tensor
    expert_indices: torch.Tensor
    token_inputs: torch.Tensor


class MoELayer(nn.Module):
    """Simple top-k Mixture-of-Experts layer with routing recorder.

    Each "expert" is a two-layer MLP (FFN). The router is a learned
    linear projection from d_model → num_experts.

    Args:
        d_model:       Input/output dimension.
        d_ffn:         Hidden dimension inside each expert FFN.
        num_experts:   Total number of experts.
        top_k:         How many experts each token is routed to.
        noise_std:     Gaussian noise added to gate logits during training
                       (helps load balancing). Set 0 to disable.
        record_routing: Whether to store RoutingRecord after each forward pass.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        record_routing: bool = True,
    ) -> None:
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be ≤ num_experts ({num_experts})")

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.record_routing = record_routing

        # Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(num_experts)
        ])

        # Populated after each forward pass when record_routing=True
        self.last_record: Optional[RoutingRecord] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               OR (batch, d_model) for token-level inputs.

        Returns:
            Output tensor, same shape as input.
        """
        original_shape = x.shape
        # Flatten to (N, d_model)
        if x.dim() == 3:
            batch, seq_len, d = x.shape
            x_flat = x.reshape(-1, d)
        else:
            x_flat = x  # already (N, d_model)

        N = x_flat.shape[0]

        # --- Router ---
        gate_logits = self.gate(x_flat)  # (N, num_experts)

        if self.training and self.noise_std > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        # Top-k selection
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # (N, top_k)

        # Softmax over selected logits only
        topk_weights = F.softmax(topk_vals, dim=-1)  # (N, top_k)

        # Build full sparse weight matrix (N, num_experts)
        gate_weights = torch.zeros(N, self.num_experts, device=x.device, dtype=x.dtype)
        gate_weights.scatter_(1, topk_idx, topk_weights)

        # --- Expert dispatch ---
        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            # Mask of tokens routed to this expert
            mask = (gate_weights[:, expert_id] > 0)  # (N,)
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = expert(expert_input)  # (n_routed, d_model)
                w = gate_weights[mask, expert_id].unsqueeze(-1)  # (n_routed, 1)
                output[mask] += w * expert_out

        # --- Record routing ---
        if self.record_routing:
            self.last_record = RoutingRecord(
                gate_logits=gate_logits.detach().clone(),
                gate_weights=gate_weights.detach().clone(),
                expert_indices=topk_idx.detach().clone(),
                token_inputs=x_flat.detach().clone(),
            )

        return output.reshape(original_shape)
