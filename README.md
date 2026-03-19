# Explainable-MoE-Debugger

**Visualization and explainability tools for Mixture-of-Experts routing decisions.**

Mixture-of-Experts (MoE) models route each token to a subset of "expert" sub-networks. When something goes wrong — dead experts, load imbalance, unexpected specialization — it's notoriously hard to debug. This library makes MoE routing transparent.

```
  Expert Utilization:
    Expert  0: ████████████████████░░░░░░░░░░░  50.3%
    Expert  1: ████████████████████░░░░░░░░░░░  50.1%
    Expert  2: ███████████████████░░░░░░░░░░░░  49.6%  [DEAD]
    Expert  3: ████████████████████░░░░░░░░░░░  50.1%

  Top Routing Features (what the router looks at per expert):
    Expert  0: x29(0.035), x31(0.034), x32(0.034)
    Expert  1: x2(0.033),  x8(0.031),  x53(0.029)
```

## Core Components

| Component | What it does |
|-----------|-------------|
| `MoELayer` | Top-k MoE layer that records routing decisions (which expert, gate weights, input) |
| `RoutingExplainer` | Gradient-based attribution: *why* was this token sent to this expert? |
| `ExpertActivationMap` | Profiles *what patterns* each expert has learned to process |
| `DeadExpertDetector` | Identifies underutilized experts over a sliding window of batches |
| `MoEDebugReport` | Structured report: routing stats, utilization, dead experts, feature breakdown |

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick Start

```python
import torch
from moe_debugger import (
    MoELayer, RoutingExplainer, ExpertActivationMap,
    DeadExpertDetector, MoEDebugReport,
)

# 1. Wrap your MoE layer (or use MoELayer directly)
layer = MoELayer(d_model=64, d_ffn=128, num_experts=8, top_k=2)

# 2. Attach debugger components
detector = DeadExpertDetector(layer, utilization_threshold=0.01)
act_map  = ExpertActivationMap(layer)

# 3. Run your model (training or eval)
for x, y in dataloader:
    out = layer(x)                  # routing recorded automatically
    detector.update()               # accumulate utilization stats
    act_map.update()                # accumulate activation profiles

# 4. Explain routing on a diagnostic batch
explainer = RoutingExplainer(layer)
layer(x_diagnostic)                 # one more forward pass

# 5. Build and print the report
report = MoEDebugReport.build(
    layer=layer,
    dead_expert_detector=detector,
    routing_explainer=explainer,
    activation_map=act_map,
    top_k_features=5,
    feature_names=[f"x{i}" for i in range(64)],  # optional
)
report.print_summary()

# Also available as JSON
print(report.to_json())
```

## Components in Detail

### `MoELayer`

A standard top-k MoE layer (linear gate → top-k selection → weighted expert sum) that stores a `RoutingRecord` after every forward pass:

```python
layer = MoELayer(d_model=256, d_ffn=1024, num_experts=8, top_k=2)
out = layer(x)              # (batch, seq_len, d_model)

rec = layer.last_record
rec.gate_weights            # (N, num_experts) — sparse softmax weights
rec.expert_indices          # (N, top_k)       — which experts were selected
rec.token_inputs            # (N, d_model)     — the inputs that were routed
```

Plug it into any model where you'd normally use an FFN block.

### `RoutingExplainer`

Answers: *which input features caused the router to send this token to expert k?*

Uses vanilla gradient attribution: `|∂ gate_logit_k / ∂ input_j|`.

```python
explainer = RoutingExplainer(layer)
attrs = explainer.compute_attribution()
# attrs[2] → (n_tokens_routed_to_expert_2, d_model) attribution tensor

top = explainer.top_features_per_expert(top_k=5)
# top[2] → [("x39", 0.043), ("x9", 0.038), ...]
```

### `ExpertActivationMap`

Answers: *what has each expert learned to process?*

Attributes mean hidden-layer activations back to input dimensions via the first-layer weight matrix.

```python
act_map = ExpertActivationMap(layer)
for x in batches:
    layer(x)
    act_map.update()        # accumulate over many batches

profiles = act_map.get_profiles()   # {expert_id: (d_model,) tensor}
top = act_map.top_activating_features(top_k=5)
```

### `DeadExpertDetector`

Tracks per-batch expert utilization in a sliding window. Flags experts below a utilization threshold as "dead" and experts receiving a disproportionate share as "overloaded".

```python
detector = DeadExpertDetector(
    layer,
    utilization_threshold=0.01,   # < 1% of tokens → dead
    overload_threshold=2.0,        # > 2× ideal share → overloaded
    window_size=50,                # batches to average over
)
for x in batches:
    layer(x)
    detector.update()

print(detector.dead_experts())      # [2, 5]
print(detector.overloaded_experts()) # [0]
```

### `MoEDebugReport`

Assembles all components into a single structured report.

```python
report = MoEDebugReport.build(layer=layer, ...)
report.print_summary()      # human-readable terminal output
report.to_json()            # JSON string
report.to_dict()            # plain dict
```

## Running the Demo

```bash
python demo.py
```

Trains a small MoE on a synthetic binary classification task, then runs the full debugger and prints a report showing routing entropy, expert utilization bars, and top routing/activation features per expert.

## Running Tests

```bash
pytest tests/ -v
```

28 tests, ~93% coverage.

## How It Works

```
Forward pass
     │
     ▼
MoELayer.forward()
  ├── gate(x) → logits (N, E)
  ├── top-k selection → expert_indices, gate_weights
  ├── dispatch tokens to experts
  ├── weighted sum of expert outputs
  └── store RoutingRecord (detached tensors)

Post-forward:
  ├── DeadExpertDetector.update()   — rolling utilization window
  └── ExpertActivationMap.update()  — activation attribution accumulation

Diagnostic pass (grad enabled):
  └── RoutingExplainer.compute_attribution()
        └── ∂ gate_logit_k / ∂ token_input → feature importance

MoEDebugReport.build()
  └── combines all of the above into a structured report
```

## License

MIT