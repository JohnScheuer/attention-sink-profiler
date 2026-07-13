# attention-sink-profiler

Empirical measurement of the **attention sink** phenomenon in autoregressive
transformers — from attention map analysis to functional impact via masked-key ablation.

## What is attention sink?

In autoregressive transformers, early tokens in the sequence (especially position 0)
receive disproportionately high attention scores regardless of their semantic content.
First documented in *Efficient Streaming Language Models with Attention Sinks*
(Xiao et al., 2023), this phenomenon has direct implications for KV-cache eviction,
streaming inference, and memory management in LLM serving systems.

This project measures it empirically on real model weights running on GPU.

---

## Models and hardware

| Model | Params | Layers | Heads | Sequence lengths |
|---|---|---|---|---|
| gpt2 | 117M | 12 | 12 | 64, 128, 256, 512, 768, 1024 |
| gpt2-medium | 345M | 24 | 16 | 64, 128, 256, 512 |

**Hardware:** NVIDIA RTX 2070 (8.6 GB VRAM)
**Stack:** PyTorch 2.13, Transformers 5.13, Python 3.14

---

## Key findings

### Finding 1 — Sink is structural, not semantic

Tested natural text vs random tokens vs repeated tokens as input.

| Input type | Mean sink_first_1 | Mean sink_first_4 |
|---|---|---|
| natural | 0.355 | 0.359 |
| **random** | **0.437** | **0.441** |
| repeated | 0.409 | 0.414 |

Random tokens produce the strongest sink. The phenomenon is positional/structural —
the model uses early positions as an attention mass dump regardless of token content.

### Finding 2 — Sink grows with depth and context length

GPT-2 mean boost over uniform baseline for first 4 tokens:

| seq_len | boost_first_4 |
|---|---|
| 64 | 2.4× |
| 128 | 9.9× |
| 256 | 22.6× |
| 512 | 45.8× |
| 768 | 67.9× |
| 1024 | **82.2×** |

Individual heads at seq=1024 reach **boost > 200×** with 90% of attention mass
concentrated on the first 4 tokens.

By layer depth (GPT-2, seq=1024):

| Layer | boost_first_4 |
|---|---|
| 0 | 0.15× |
| 2 | 18.6× |
| 5 | **118.2×** |
| 7 | **132.7×** |
| 9 | **130.1×** |
| 11 | **119.9×** |

Nearly absent in early layers, dominant in mid-to-late layers.

### Finding 3 — Larger models dilute the sink

| Model | seq=512 sink_share_first_4 | boost_first_4 |
|---|---|---|
| gpt2 | 0.382 | 45.8× |
| gpt2-medium | **0.199** | **23.8×** |

GPT-2-medium distributes attention more broadly at longer contexts.
The sink phenomenon dilutes with model scale.

### Finding 4 — Per-head classification

At seq=512, GPT-2 heads classified by attention pattern:

| Head type | Count | Fraction |
|---|---|---|
| strong_sink | 46 | 32% |
| moderate_sink | 40 | 28% |
| distributed | 32 | 22% |
| recency | 14 | 10% |
| mixed | 12 | 8% |

60% of all heads (86/144) are sink heads. The effect concentrates in deeper layers:
- Layers 0–1: 0 strong sink heads
- Layer 8: **9/12** strong sink
- Layer 9: **8/12** strong sink

### Finding 5 — Functional impact via masked-key ablation

The most rigorous experiment: mask specific key positions in the attention
computation while keeping the full sequence and correct positional embeddings.

**Tail perplexity ratio** (higher = more degradation) at seq=1024:

| Masked region | Size | PPL ratio | Top-5 overlap |
|---|---|---|---|
| recent_4 | 4 | **1.868** | 100% |
| recent_8 | 8 | **1.311** | 100% |
| **first_8** | 8 | **1.075** | **40%** |
| random_4_mean | 4 | 1.043 | 100% |
| **first_4** | 4 | **1.017** | **40%** |
| first_1 | 1 | 1.015 | **40%** |
| middle_8 | 8 | 1.005 | 100% |
| middle_4 | 4 | 1.000 | 100% |
| random_8_mean | 8 | 1.000 | 93% |

**Interpretation:**

1. **Recent tokens are the strongest functional dependency** for immediate next-token
   prediction — masking the last 4 tokens before the evaluation window causes 87%
   perplexity increase.

2. **Sink tokens (first 8) matter more than middle or random tokens** — masking
   first_8 degrades perplexity by 7.5%, while middle_8 causes only 0.5%.

3. **Sink tokens strongly perturb the output distribution** — top-5 overlap drops
   to 40% when first tokens are masked, while middle/random masks maintain 93–100%.
   This means sink tokens have a directed, discrete impact on which tokens the model
   considers most likely, even when average tail loss changes modestly.

**Conclusion:** attention sink tokens are not just visually salient in attention maps —
they are functionally important. However, their importance is qualitatively different
from recency: they affect output *distribution shape* more than average *loss magnitude*.

---

## Experiments

| # | Experiment | Script | Key output |
|---|---|---|---|
| Main | Attention map profiling | profile_attention_sink.py | attention_sink_summary.csv |
| 1 | Natural vs random vs repeated | advanced_experiments.py | experiment1_input_types.csv |
| 3 | Per-head sink classification | advanced_experiments.py | experiment3_head_classification.csv |
| 2g | Masked-key ablation | experiment2g_masked_key_ablation.py | experiment2g_masked_key_ablation.csv |

Experiments 2c–2f explored KV eviction via input truncation and positional remapping.
These were informative but architecturally limited: GPT-2 uses absolute positional
embeddings, which makes sparse-context retention unstable. Experiment 2g (masked-key
ablation) provides the correct functional measurement by keeping the full sequence
and positions intact while blocking attention to specific keys.

---

## Results files

| File | Rows | Description |
|---|---|---|
| attention_sink_summary.csv | 2400 | Per-layer, per-head sink metrics (both models) |
| attention_sink_positions.csv | 56064 | Per-position attention mass by layer |
| experiment1_input_types.csv | 9 | Natural vs random vs repeated |
| experiment3_head_classification.csv | 144 | Head-level classification |
| experiment2g_masked_key_ablation.csv | ~48 | Per-mask tail PPL and KL divergence |
| experiment2g_masked_key_ablation_agg.csv | ~30 | Aggregated with random trial means |

---

## Plots

| File | Description |
|---|---|
| gpt2_sink_share_first4_by_layer.png | Sink share across layers |
| gpt2_boost_first4_by_layer.png | Boost over uniform by layer |
| gpt2_heatmap_seq*.png | Attention received heatmaps |
| gpt2-medium_*.png | Same plots for GPT-2-medium |
| model_comparison_by_layer.png | GPT-2 vs GPT-2-medium by depth |
| exp1_input_types.png | Natural vs random vs repeated |
| exp3_head_classification.png | Head type distribution + heatmap |
| exp2g_ppl_ratio.png | Masked-key PPL ratio |
| exp2g_grouped_ppl_ratio.png | Grouped comparison: sink vs middle vs recent |

---

## Run

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    # Main attention map profiling
    python3 profile_attention_sink.py \
        --models gpt2 --seq-lens 64 128 256 512 768 1024
    python3 profile_attention_sink.py \
        --models gpt2-medium --seq-lens 64 128 256 512

    # Generate plots
    python3 plot_attention_sink.py

    # Advanced experiments (input types, head classification)
    python3 advanced_experiments.py

    # Masked-key ablation (functional impact)
    python3 experiment2g_masked_key_ablation.py

---

## References

- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
  — https://arxiv.org/abs/2309.17453
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) — GPT-2
