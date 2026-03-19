#!/usr/bin/env python3
"""
demo.py - Explainable MoE Debugger demonstration.

Trains a small MoE model on a synthetic task, then runs the full
debugger suite and prints a report.

Synthetic task:
  Input x ∈ R^64.  y = sum(x[:32]) > sum(x[32:])  (binary classification).
  The hypothesis is that experts should specialize on different halves
  of the input.  The debugger should reveal this specialization.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from repo root
sys.path.insert(0, "src")

from moe_debugger import (
    MoELayer,
    RoutingExplainer,
    ExpertActivationMap,
    DeadExpertDetector,
    MoEDebugReport,
)


# ---------------------------------------------------------------------------
# Tiny classifier that wraps a MoELayer
# ---------------------------------------------------------------------------

class MoEClassifier(nn.Module):
    def __init__(self, d_model=64, d_ffn=128, num_experts=4, top_k=2):
        super().__init__()
        self.moe = MoELayer(
            d_model=d_model,
            d_ffn=d_ffn,
            num_experts=num_experts,
            top_k=top_k,
            record_routing=True,
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.moe(x)         # (batch, d_model)
        return self.head(h)     # (batch, 2)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_batch(n=256, d=64, device="cpu"):
    """Binary classification: y = 1 if sum(x[:32]) > sum(x[32:])."""
    x = torch.randn(n, d, device=device)
    y = (x[:, :32].sum(dim=1) > x[:, 32:].sum(dim=1)).long()
    return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, n_steps=300, batch_size=256, lr=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(n_steps):
        x, y = make_batch(batch_size, device=device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"  step {step+1:4d}/{n_steps}  loss={loss.item():.4f}  acc={acc:.3f}")


# ---------------------------------------------------------------------------
# Debug pass - collect stats over multiple batches
# ---------------------------------------------------------------------------

def collect_debug_stats(model, n_batches=50, batch_size=256, device="cpu"):
    dead_detector = DeadExpertDetector(
        model.moe,
        utilization_threshold=0.02,
        window_size=n_batches,
    )
    act_map = ExpertActivationMap(model.moe)

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = make_batch(batch_size, device=device)
            model(x)
            dead_detector.update()
            act_map.update()

    return dead_detector, act_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = "cpu"

    D = 64
    feature_names = [f"x{i}" for i in range(D)]

    print("=" * 60)
    print("  Explainable MoE Debugger - Demo")
    print("=" * 60)
    print()
    print("Building MoE classifier (4 experts, top-2 routing)...")
    model = MoEClassifier(d_model=D, d_ffn=128, num_experts=4, top_k=2)
    model.to(device)

    print(f"\nTraining on synthetic task (d={D}, task: sum(x[:32]) > sum(x[32:]))...")
    train(model, n_steps=300, device=device)

    # Final accuracy
    model.eval()
    with torch.no_grad():
        x_test, y_test = make_batch(1024, device=device)
        logits = model(x_test)
        final_acc = (logits.argmax(-1) == y_test).float().mean().item()
    print(f"\n  Final test accuracy: {final_acc:.3f}")

    # ---- Run explainer on one fresh batch ----
    print("\nRunning debugger components...")
    x_debug, _ = make_batch(512, device=device)

    # Enable grad for explainer
    x_debug_req = x_debug.clone().detach()
    with torch.no_grad():
        model(x_debug_req)   # populates last_record

    explainer = RoutingExplainer(model.moe)

    # Collect activation + dead-expert stats over 50 batches
    dead_detector, act_map = collect_debug_stats(model, n_batches=50, device=device)

    # Re-run one pass so last_record is fresh for the report
    x_final, _ = make_batch(512, device=device)
    with torch.no_grad():
        model(x_final)
    dead_detector.update()
    act_map.update()

    # ---- Build report ----
    report = MoEDebugReport.build(
        layer=model.moe,
        dead_expert_detector=dead_detector,
        routing_explainer=explainer,
        activation_map=act_map,
        top_k_features=5,
        feature_names=feature_names,
    )

    report.print_summary()

    print("\nFull report (JSON snippet):")
    import json
    d = report.to_dict()
    # Print a compact version (no raw feature lists)
    compact = {
        "layer_info": d["layer_info"],
        "routing_entropy": d["routing_entropy"],
        "load_imbalance": d["load_imbalance"],
        "dead_experts": d["dead_experts"],
        "overloaded_experts": d["overloaded_experts"],
        "raw_utilization": d["raw_utilization"],
    }
    print(json.dumps(compact, indent=2))

    return report


if __name__ == "__main__":
    main()
