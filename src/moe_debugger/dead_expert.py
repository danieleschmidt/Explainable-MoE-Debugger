"""DeadExpertDetector: identifies underutilized / collapsed experts.

A "dead" expert is one that consistently receives fewer tokens than a
utilization threshold over a sliding window of batches.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

from .layer import MoELayer


@dataclass
class ExpertUtilizationStats:
    """Aggregated utilization stats for one expert."""
    expert_id: int
    total_tokens_seen: int
    total_tokens_routed: int
    utilization_rate: float          # fraction of tokens routed here overall
    batch_utilization_history: List[float] = field(default_factory=list)
    is_dead: bool = False
    is_overloaded: bool = False


class DeadExpertDetector:
    """Tracks expert utilization across batches and flags dead / overloaded experts.

    Args:
        layer:             The MoELayer to monitor.
        utilization_threshold: An expert is "dead" if its mean utilization
                            rate over the window falls below this value.
                            E.g. 0.01 means less than 1% of tokens routed to it.
        overload_threshold: An expert is "overloaded" if it receives more than
                            this fraction of tokens (ideal = 1/num_experts).
        window_size:       How many recent batches to consider for rolling stats.
    """

    def __init__(
        self,
        layer: MoELayer,
        utilization_threshold: float = 0.01,
        overload_threshold: float = 2.0,  # relative to 1/num_experts
        window_size: int = 50,
    ) -> None:
        self.layer = layer
        self.utilization_threshold = utilization_threshold
        self.overload_threshold = overload_threshold
        self.window_size = window_size

        # Per-expert sliding window of per-batch utilization rates
        self._windows: Dict[int, Deque[float]] = {
            i: deque(maxlen=window_size) for i in range(layer.num_experts)
        }
        self._total_tokens: int = 0
        self._expert_tokens: Dict[int, int] = {i: 0 for i in range(layer.num_experts)}

    def reset(self) -> None:
        """Reset all accumulated state."""
        for d in self._windows.values():
            d.clear()
        self._total_tokens = 0
        self._expert_tokens = {i: 0 for i in range(self.layer.num_experts)}

    def update(self) -> None:
        """Process the most recent forward pass from the MoELayer.

        Call after each forward pass to accumulate statistics.
        """
        record = self.layer.last_record
        if record is None:
            raise RuntimeError(
                "No routing record found. Run a forward pass through the MoELayer first."
            )

        n_tokens = record.gate_weights.shape[0]
        self._total_tokens += n_tokens

        for expert_id in range(self.layer.num_experts):
            n_routed = int((record.gate_weights[:, expert_id] > 0).sum().item())
            self._expert_tokens[expert_id] += n_routed
            batch_rate = n_routed / max(n_tokens, 1)
            # Always append — including 0.0 for completely skipped experts
            self._windows[expert_id].append(batch_rate)

    def get_stats(self) -> List[ExpertUtilizationStats]:
        """Return per-expert utilization statistics.

        Returns:
            List of ExpertUtilizationStats, one per expert.
        """
        ideal_rate = self.layer.top_k / self.layer.num_experts
        overload_abs = ideal_rate * self.overload_threshold

        stats = []
        for expert_id in range(self.layer.num_experts):
            history = list(self._windows[expert_id])
            overall_rate = (
                self._expert_tokens[expert_id] / max(self._total_tokens, 1)
            )
            rolling_rate = sum(history) / max(len(history), 1)

            is_dead = rolling_rate < self.utilization_threshold and len(history) > 0
            is_overloaded = rolling_rate > overload_abs and len(history) > 0

            stats.append(ExpertUtilizationStats(
                expert_id=expert_id,
                total_tokens_seen=self._total_tokens,
                total_tokens_routed=self._expert_tokens[expert_id],
                utilization_rate=overall_rate,
                batch_utilization_history=history,
                is_dead=is_dead,
                is_overloaded=is_overloaded,
            ))

        return stats

    def dead_experts(self) -> List[int]:
        """Return list of expert IDs classified as dead."""
        return [s.expert_id for s in self.get_stats() if s.is_dead]

    def overloaded_experts(self) -> List[int]:
        """Return list of expert IDs classified as overloaded."""
        return [s.expert_id for s in self.get_stats() if s.is_overloaded]
