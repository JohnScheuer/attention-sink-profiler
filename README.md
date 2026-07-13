# attention-sink-profiler

## Overview

Empirical measurement of the **attention sink** phenomenon in autoregressive
transformers — from attention map analysis to functional impact via
**masked-key ablation**.

This project studies how early tokens in a sequence absorb disproportionate
attention mass, how that effect changes with depth and sequence length, and
whether those sink tokens matter functionally for prediction quality.

---

## What is Attention Sink?

Attention sink is the tendency of autoregressive transformers to assign
disproportionately high attention mass to the earliest tokens in a sequence,
often independent of semantic content.

This phenomenon matters for:

- **KV-cache eviction**
- **streaming inference**
- **memory-efficient decoding**
- **attention interpretability**

Reference:
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2023)

---

## Models and Hardware

### Models

| Model | Params | Layers | Heads | Sequence lengths |
|---|---:|---:|---:|---|
| gpt2 | 117M | 12 | 12 | 64, 128, 256, 512, 768, 1024 |
| gpt2-medium | 345M | 24 | 16 | 64, 128, 256, 512 |

### Hardware

- **GPU:** NVIDIA RTX 2070
- **VRAM:** 8.6 GB
- **Framework:** PyTorch 2.13
- **Library:** Transformers 5.13
- **Python:** 3.14

---

## Research Questions

1. Is attention sink semantic or structural?
2. How does sink strength change with depth?
3. How does sink strength scale with sequence length?
4. Do larger models behave differently?
5. Are sink tokens functionally important?

---

## Methodology

### Main Profiling

For each `(model, seq_len)` pair, the pipeline:

1. builds a sequence of target length
2. runs a forward pass with `output_attentions=True`
3. extracts attention matrices for all layers and heads
4. computes sink metrics on the final query positions

### Core Metrics

- **sink_share_first_1 / 4 / 8**  
  Fraction of attention mass assigned to the first 1, 4, or 8 tokens.

- **boost_first_1 / 4 / 8**  
  Observed sink share divided by the expected uniform baseline.

- **peak_key_pos**  
  Position receiving the highest attention mass.

- **top1_first4_query_frac / top1_first8_query_frac**  
  Fraction of tail queries whose most-attended key lies in the first 4 or 8 tokens.

### Tail Window

Metrics are computed over the last `tail_window` query positions
(default: 64) to avoid trivial early-position effects from the causal mask.

---

## Experiments

### 1. Input Type Ablation

#### Goal
Test whether sink depends on semantic content.

#### Inputs
- natural text
- random tokens
- repeated token

#### Result

| Input type | Mean sink_first_1 | Mean sink_first_4 |
|---|---:|---:|
| natural | 0.355 | 0.359 |
| **random** | **0.437** | **0.441** |
| repeated | 0.409 | 0.414 |

#### Interpretation
Random tokens produce the strongest sink.

This suggests the phenomenon is **structural/positional**, not semantic.

---

### 2. Main Attention Sink Sweep

#### GPT-2: boost over uniform for first 4 tokens

| seq_len | boost_first_4 |
|---|---:|
| 64 | 2.4× |
| 128 | 9.9× |
| 256 | 22.6× |
| 512 | 45.8× |
| 768 | 67.9× |
| 1024 | **82.2×** |

#### Interpretation
Sink strength grows strongly with sequence length.

At long context, the first 4 tokens absorb vastly more attention than a uniform
baseline would predict.

---

### 3. Layer Depth Analysis

#### GPT-2 at seq_len = 1024

| Layer | boost_first_4 |
|---|---:|
| 0 | 0.15× |
| 2 | 18.6× |
| 5 | 118.2× |
| 7 | 132.7× |
| 9 | 130.1× |
| 11 | 119.9× |

#### Interpretation
The sink is nearly absent in early layers and becomes dominant in
mid-to-late layers.

---

### 4. GPT-2 vs GPT-2 Medium

#### At seq_len = 512

| Model | sink_share_first_4 | boost_first_4 |
|---|---:|---:|
| gpt2 | 0.382 | 45.8× |
| gpt2-medium | **0.199** | **23.8×** |

#### Interpretation
The larger model spreads attention more broadly.

Sink is still present, but less concentrated.

---

### 5. Per-Head Classification

#### Head Types

| Head type | Count | Fraction |
|---|---:|---:|
| strong_sink | 46 | 32% |
| moderate_sink | 40 | 28% |
| distributed | 32 | 22% |
| recency | 14 | 10% |
| mixed | 12 | 8% |

#### Interpretation
About **60% of GPT-2 heads** are sink-oriented (`strong_sink` or `moderate_sink`).

The effect is concentrated in deeper layers:
- layers 0–1: no strong sink heads
- layer 8: **9 / 12** strong sink heads
- layer 9: **8 / 12** strong sink heads

---

### 6. Masked-Key Ablation

#### Goal
Measure the **functional impact** of sink tokens.

#### Why this experiment matters
Naive token removal creates positional artifacts in GPT-2 because it uses
**absolute positional embeddings**.

Instead, this experiment:

- keeps the full sequence
- keeps normal positional structure
- blocks attention to selected key positions
- measures degradation in tail prediction quality

#### Mask types
- `first_1`, `first_4`, `first_8`
- `middle_4`, `middle_8`
- `recent_4`, `recent_8`
- `random_4`, `random_8`

#### Key result at seq_len = 1024

| Masked region | Size | PPL ratio | Top-5 overlap |
|---|---:|---:|---:|
| recent_4 | 4 | **1.868** | 100% |
| recent_8 | 8 | **1.311** | 100% |
| **first_8** | 8 | **1.075** | **40%** |
| random_4_mean | 4 | 1.043 | 100% |
| **first_4** | 4 | **1.017** | **40%** |
| first_1 | 1 | 1.015 | **40%** |
| middle_8 | 8 | 1.005 | 100% |
| middle_4 | 4 | 1.000 | 100% |
| random_8_mean | 8 | 1.000 | 93% |

#### Interpretation

- **Recent tokens are the strongest functional dependency** for immediate next-token prediction.
- **Sink tokens still matter more than middle or random tokens**.
- **Sink masking strongly perturbs output ranking**, even when average tail loss changes modestly.

#### Conclusion
Sink tokens are not just visually salient in attention maps — they are
**functionally important**, especially for output distribution shape.

---

## Main Findings

1. Attention sink is **structural**, not semantic.
2. Sink grows with **layer depth**.
3. Sink becomes more extreme as **sequence length increases**.
4. Larger models like GPT-2 medium show a **weaker, more distributed sink**.
5. Sink tokens have measurable **functional impact**, but **recent tokens remain the strongest dependency** for immediate next-token prediction.

---

## Results Files

| File | Description |
|---|---|
| `results/attention_sink_summary.csv` | Per-layer, per-head sink metrics |
| `results/attention_sink_positions.csv` | Per-position attention mass by layer |
| `results/experiment1_input_types.csv` | Natural vs random vs repeated |
| `results/experiment3_head_classification.csv` | Head-level classification |
| `results/experiment2g_masked_key_ablation.csv` | Functional masked-key ablation |
| `results/experiment2g_masked_key_ablation_agg.csv` | Aggregated random-trial results |
| `results/run_log.csv` | Timing and run status |
| `results/metadata.json` | Configuration metadata |

---

## Plots

| File | Description |
|---|---|
| `plots/gpt2_sink_share_first4_by_layer.png` | Sink share across layers |
| `plots/gpt2_boost_first4_by_layer.png` | Boost over uniform by layer |
| `plots/gpt2_heatmap_seq*.png` | Attention-received heatmaps |
| `plots/gpt2-medium_*.png` | Same plots for GPT-2 medium |
| `plots/model_comparison_by_layer.png` | GPT-2 vs GPT-2 medium |
| `plots/exp1_input_types.png` | Input type ablation |
| `plots/exp3_head_classification.png` | Head type distribution |
| `plots/exp2g_ppl_ratio.png` | PPL ratio under masked-key ablation |
| `plots/exp2g_grouped_ppl_ratio.png` | Sink vs middle vs recent vs random |

---

## Repository Structure

~~~text
attention-sink-profiler/
├── profile_attention_sink.py
├── plot_attention_sink.py
├── advanced_experiments.py
├── experiment2g_masked_key_ablation.py
├── README.md
├── DESIGN.md
├── LICENSE
├── requirements.txt
├── results/
└── plots/
~~~

---

## How to Run

### 1. Create environment

~~~bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
~~~

### 2. Run main attention profiling

~~~bash
python3 profile_attention_sink.py --models gpt2 --seq-lens 64 128 256 512 768 1024
python3 profile_attention_sink.py --models gpt2-medium --seq-lens 64 128 256 512
~~~

### 3. Generate plots

~~~bash
python3 plot_attention_sink.py
~~~

### 4. Run advanced experiments

~~~bash
python3 advanced_experiments.py
~~~

### 5. Run masked-key ablation

~~~bash
python3 experiment2g_masked_key_ablation.py
~~~

---

## Limitations

### GPT-2 uses absolute positional embeddings
This makes direct simulation of KV eviction tricky: removing tokens can create
positional artifacts not representative of modern RoPE-based models.

### Masked-key ablation is the cleanest functional test here
It avoids sparse-position artifacts by keeping the full sequence intact and
only blocking access to selected keys.

### Results may differ on modern architectures
A RoPE-based model such as LLaMA or Mistral could show different streaming/eviction behavior.

---

## References

- Xiao et al., **Efficient Streaming Language Models with Attention Sinks** (2023)  
  https://arxiv.org/abs/2309.17453

- Radford et al., **Language Models are Unsupervised Multitask Learners** (2019)  
  GPT-2 technical report

