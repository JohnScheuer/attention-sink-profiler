# Design — attention-sink-profiler

## Objective

Empirically measure the attention sink phenomenon in autoregressive transformers
and determine whether it has functional impact beyond visual salience in attention maps.

## Architecture

    ┌──────────────────────────────────────────────┐
    │              Input Construction              │
    │  natural text / random tokens / repeated     │
    │  tokenized and truncated to target seq_len   │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │           Forward Pass (eager attn)          │
    │  output_attentions=True, use_cache=False     │
    │  returns attention weights per layer/head    │
    └───────────────────┬──────────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
    ┌──────────────────┐  ┌──────────────────────┐
    │  Attention Map    │  │   Masked-Key         │
    │  Analysis         │  │   Ablation           │
    │                   │  │                      │
    │  • sink share     │  │  • attention_mask     │
    │  • boost over     │  │    blocks keys        │
    │    uniform        │  │  • measures tail PPL  │
    │  • peak position  │  │  • KL divergence     │
    │  • head classify  │  │  • top-K overlap     │
    └────────┬──────────┘  └──────────┬───────────┘
             │                        │
             ▼                        ▼
    ┌──────────────────────────────────────────────┐
    │               CSV Results                    │
    │  attention_sink_summary.csv                  │
    │  attention_sink_positions.csv                │
    │  experiment1_input_types.csv                 │
    │  experiment3_head_classification.csv          │
    │  experiment2g_masked_key_ablation.csv         │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │               Visualization                  │
    │  matplotlib heatmaps, line plots, bar charts │
    └──────────────────────────────────────────────┘

## Methodology

### Main profiling (profile_attention_sink.py)

For each (model, seq_len) combination:

1. Build input_ids of target length from repeated natural text
2. Run forward pass with `output_attentions=True`
3. Extract attention tensors: `[layers][batch, heads, q, k]`
4. For each layer and head, compute:
   - **sink_share_first_K**: fraction of attention mass (from tail queries)
     going to the first K key positions
   - **boost_first_K**: sink_share / expected_uniform_share
   - **peak_key_pos**: which key position receives the most attention
   - **top1_first_K_frac**: fraction of tail queries whose argmax key
     is in the first K positions

### Tail window

Metrics are computed over the last `tail_window` query positions (default: 64).
This avoids measuring attention patterns at positions where the causal mask
naturally forces attention onto early tokens (e.g., position 2 can only attend
to positions 0, 1, 2 — high concentration on early tokens is trivial there).

### Uniform baseline

For each query position q, the expected uniform share of the first K tokens is:

    uniform(q, K) = min(K, q+1) / (q+1)

The boost metric normalizes the observed sink share against this baseline,
making results comparable across sequence lengths.

## Experiment design

### Experiment 1 — Input type ablation

Tests whether the sink depends on semantic content:
- **natural**: repeated coherent English text
- **random**: uniformly sampled token IDs
- **repeated**: single token repeated seq_len times

If random >= natural, the sink is positional, not semantic.

### Experiment 3 — Head classification

Classifies each head into one of five types based on its attention pattern:

| Type | Criterion |
|---|---|
| strong_sink | sink_share_first_4 > 0.5 |
| moderate_sink | sink_share_first_4 > 0.2 |
| distributed | normalized entropy > 0.8 |
| recency | peak position > 80% of seq_len |
| mixed | none of the above |

### Experiment 2g — Masked-key ablation

The most rigorous functional test. Instead of removing tokens from the input
(which creates positional artifacts with absolute PE), this experiment:

1. Keeps the full input sequence and position IDs unchanged
2. Uses `attention_mask` to block access to specific key positions
3. Measures tail perplexity and last-token distribution changes

Mask types compared:
- **first_K**: sink tokens (positions 0..K-1)
- **middle_K**: tokens at center of sequence
- **recent_K**: tokens just before the evaluation window
- **random_K**: randomly placed windows (3 trials, averaged)

This directly approximates KV-cache eviction without positional artifacts.

## Key design decisions

### Why eager attention?
Modern transformers default to FlashAttention or SDPA, which don't return
attention weight matrices. We force `attn_implementation="eager"` to get
the full `[heads, q, k]` attention tensors needed for analysis.

### Why GPT-2?
- Small enough to fit on consumer GPUs with full attention matrices
- Well-studied architecture with known attention patterns
- Absolute positional embeddings create an interesting contrast with
  RoPE-based models (where StreamingLLM was originally tested)
- Available in multiple sizes (small, medium) for scale comparison

### Why not remove tokens from input for eviction tests?
GPT-2 uses absolute positional embeddings. Removing tokens and re-indexing
positions (0..budget-1) changes the positional signal. Keeping original
positions but with gaps creates sequences the model never saw in training.
Both approaches produce artifacts. The attention_mask approach avoids this
by keeping positions intact and only blocking key access.

### Memory management
Attention tensors for large sequences are expensive:
- seq=1024, 12 layers, 12 heads: ~600 MB in float32
- We detach and move to CPU immediately after extraction
- Explicit `torch.cuda.empty_cache()` between runs
- Models are deleted between model switches

## File structure

    attention-sink-profiler/
    ├── profile_attention_sink.py         # Main attention map profiling
    ├── plot_attention_sink.py            # Visualization for main results
    ├── advanced_experiments.py           # Exp 1 (input types) + Exp 3 (head classification)
    ├── experiment2g_masked_key_ablation.py  # Exp 2g (functional impact)
    ├── requirements.txt
    ├── README.md
    ├── DESIGN.md
    ├── LICENSE
    ├── results/                          # CSV outputs
    │   ├── attention_sink_summary.csv
    │   ├── attention_sink_positions.csv
    │   ├── experiment1_input_types.csv
    │   ├── experiment3_head_classification.csv
    │   ├── experiment2g_masked_key_ablation.csv
    │   ├── experiment2g_masked_key_ablation_agg.csv
    │   ├── run_log.csv
    │   └── metadata.json
    └── plots/                            # PNG outputs
        ├── gpt2_sink_share_first4_by_layer.png
        ├── gpt2_boost_first4_by_layer.png
        ├── gpt2_heatmap_seq*.png
        ├── gpt2-medium_*.png
        ├── model_comparison_by_layer.png
        ├── exp1_input_types.png
        ├── exp3_head_classification.png
        ├── exp2g_ppl_ratio.png
        └── exp2g_grouped_ppl_ratio.png
