"""
Advanced attention sink experiments:
1. Random vs natural text (positional vs semantic)
2. KV ablation (practical impact of removing sink tokens)
3. Per-head sink classification
"""

import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NATURAL_TEXT = (
    "In a long technical discussion, researchers analyze language models, "
    "attention patterns, memory behavior, token generation, and inference efficiency. "
    "They compare latency, throughput, cache policies, and routing effects across "
    "different workloads while carefully measuring the contribution of each token "
    "position to the final output distribution. "
)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        )
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    model.config.output_attentions = True
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


def build_ids(tokenizer, seq_len, mode="natural"):
    if mode == "natural":
        chunk = tokenizer(NATURAL_TEXT, return_tensors="pt",
                          add_special_tokens=False).input_ids[0]
        parts = []
        total = 0
        while total < seq_len:
            parts.append(chunk)
            total += chunk.numel()
        ids = torch.cat(parts)[:seq_len]
    elif mode == "random":
        vocab_size = tokenizer.vocab_size
        ids = torch.randint(100, vocab_size - 100, (seq_len,))
    elif mode == "repeated":
        # Same token repeated — controls for content effects
        token_id = tokenizer.encode("the", add_special_tokens=False)[0]
        ids = torch.full((seq_len,), token_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ids.unsqueeze(0).to(DEVICE)


def get_attentions(model, input_ids):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attn = [a.detach().cpu() for a in out.attentions if a is not None]
    del out
    torch.cuda.empty_cache()
    return attn


def sink_metrics(attentions, tail_window=64):
    """Compute mean attention mass on first-K tokens from tail queries."""
    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]
    tail = min(tail_window, seq_len)
    q_start = seq_len - tail

    results = {}
    for k in [1, 4, 8]:
        masses = []
        for layer_attn in attentions:
            attn = layer_attn[0].float()  # [heads, q, k]
            tail_attn = attn[:, q_start:, :seq_len]
            mass = tail_attn[:, :, :min(k, seq_len)].sum(dim=-1).mean().item()
            masses.append(mass)
        results[f"sink_first_{k}"] = float(np.mean(masses))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Random vs Natural vs Repeated
# ═══════════════════════════════════════════════════════════════════════════
def experiment_1_input_types(model_name="gpt2", seq_lens=None):
    if seq_lens is None:
        seq_lens = [128, 256, 512]

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Random vs Natural vs Repeated tokens")
    print("=" * 60)

    tokenizer, model = load_model(model_name)
    rows = []

    for seq_len in seq_lens:
        for mode in ["natural", "random", "repeated"]:
            print(f"  {model_name} seq={seq_len} mode={mode}")
            ids = build_ids(tokenizer, seq_len, mode=mode)
            attentions = get_attentions(model, ids)
            metrics = sink_metrics(attentions, tail_window=64)

            row = {
                "model": model_name,
                "seq_len": seq_len,
                "input_type": mode,
            }
            row.update(metrics)
            rows.append(row)

            del attentions, ids
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "experiment1_input_types.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print(df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5 * len(seq_lens), 5),
                             sharey=True)
    if len(seq_lens) == 1:
        axes = [axes]

    for ax, sl in zip(axes, seq_lens):
        sub = df[df["seq_len"] == sl]
        x = sub["input_type"]
        ax.bar(x, sub["sink_first_1"], width=0.25, label="first 1", align="center")
        ax.bar([xi + 0.25 for xi in range(len(x))],
               sub["sink_first_4"].values, width=0.25, label="first 4")
        ax.bar([xi + 0.5 for xi in range(len(x))],
               sub["sink_first_8"].values, width=0.25, label="first 8")
        ax.set_title(f"seq_len={sl}")
        ax.set_xlabel("Input type")
        ax.set_ylabel("Mean attention mass")
        ax.set_xticks([i + 0.25 for i in range(len(x))])
        ax.set_xticklabels(x.values)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Attention sink: natural vs random vs repeated", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_input_types.png"), dpi=180,
                bbox_inches="tight")
    plt.close()
    print(f"Plot: plots/exp1_input_types.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: KV Ablation
# ═══════════════════════════════════════════════════════════════════════════
def experiment_2_kv_ablation(model_name="gpt2", seq_len=256):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: KV Ablation — impact of removing first N tokens")
    print("=" * 60)

    tokenizer, model = load_model(model_name)

    # Generate full sequence logits as baseline
    ids = build_ids(tokenizer, seq_len, mode="natural")

    with torch.no_grad():
        baseline_out = model(ids, use_cache=False, return_dict=True)
        baseline_logits = baseline_out.logits[0, -1, :].float().cpu()
        baseline_probs = torch.softmax(baseline_logits, dim=0)

    ablation_sizes = [0, 1, 2, 4, 8, 16, 32]
    rows = []

    for n_remove in ablation_sizes:
        if n_remove >= seq_len:
            continue

        if n_remove == 0:
            ablated_ids = ids
        else:
            ablated_ids = ids[:, n_remove:]

        with torch.no_grad():
            ablated_out = model(ablated_ids, use_cache=False, return_dict=True)
            ablated_logits = ablated_out.logits[0, -1, :].float().cpu()
            ablated_probs = torch.softmax(ablated_logits, dim=0)

        # KL divergence: KL(baseline || ablated)
        kl_div = torch.sum(
            baseline_probs * (torch.log(baseline_probs + 1e-10) -
                              torch.log(ablated_probs + 1e-10))
        ).item()

        # Top-1 agreement
        top1_match = int(baseline_logits.argmax().item() ==
                         ablated_logits.argmax().item())

        # Top-5 agreement
        baseline_top5 = set(baseline_logits.topk(5).indices.tolist())
        ablated_top5 = set(ablated_logits.topk(5).indices.tolist())
        top5_overlap = len(baseline_top5 & ablated_top5) / 5.0

        # Cosine similarity of logit vectors
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_logits.unsqueeze(0),
            ablated_logits.unsqueeze(0),
        ).item()

        row = {
            "model": model_name,
            "seq_len": seq_len,
            "tokens_removed": n_remove,
            "remaining_tokens": seq_len - n_remove,
            "kl_divergence": kl_div,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "cosine_similarity": cos_sim,
        }
        rows.append(row)
        print(f"  remove={n_remove:3d}  KL={kl_div:.4f}  top1={'✓' if top1_match else '✗'}  "
              f"top5={top5_overlap:.1%}  cos={cos_sim:.4f}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "experiment2_kv_ablation.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(df["tokens_removed"], df["kl_divergence"], "o-", color="#E74C3C")
    axes[0].set_title("KL Divergence from baseline")
    axes[0].set_xlabel("Tokens removed from start")
    axes[0].set_ylabel("KL(baseline || ablated)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["tokens_removed"], df["top5_overlap"], "s-", color="#2ECC71")
    axes[1].set_title("Top-5 token overlap with baseline")
    axes[1].set_xlabel("Tokens removed from start")
    axes[1].set_ylabel("Overlap fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["tokens_removed"], df["cosine_similarity"], "^-", color="#4C9BE8")
    axes[2].set_title("Cosine similarity of logits vs baseline")
    axes[2].set_xlabel("Tokens removed from start")
    axes[2].set_ylabel("Cosine similarity")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} seq={seq_len} — Effect of removing first N tokens from KV",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp2_kv_ablation.png"), dpi=180,
                bbox_inches="tight")
    plt.close()
    print(f"Plot: plots/exp2_kv_ablation.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Per-head sink classification
# ═══════════════════════════════════════════════════════════════════════════
def experiment_3_head_classification(model_name="gpt2", seq_len=512):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Per-head sink classification")
    print("=" * 60)

    tokenizer, model = load_model(model_name)
    ids = build_ids(tokenizer, seq_len, mode="natural")
    attentions = get_attentions(model, ids)

    tail = min(64, seq_len)
    q_start = seq_len - tail
    rows = []

    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn[0].float()  # [heads, q, k]
        heads = attn.shape[0]

        for head_idx in range(heads):
            head_attn = attn[head_idx, q_start:, :seq_len]  # [tail, k]
            mean_received = head_attn.mean(dim=0)  # [k]

            sink_1 = mean_received[:1].sum().item()
            sink_4 = mean_received[:min(4, seq_len)].sum().item()

            # Entropy of the mean distribution
            p = mean_received + 1e-10
            p = p / p.sum()
            entropy = -(p * p.log()).sum().item()
            max_entropy = np.log(seq_len)

            # Peak position
            peak_pos = int(mean_received.argmax().item())

            # Classify
            if sink_4 > 0.5:
                head_type = "strong_sink"
            elif sink_4 > 0.2:
                head_type = "moderate_sink"
            elif entropy / max_entropy > 0.8:
                head_type = "distributed"
            elif peak_pos > seq_len * 0.8:
                head_type = "recency"
            else:
                head_type = "mixed"

            rows.append({
                "model": model_name,
                "seq_len": seq_len,
                "layer": layer_idx,
                "head": head_idx,
                "sink_share_first_1": sink_1,
                "sink_share_first_4": sink_4,
                "entropy": entropy,
                "normalized_entropy": entropy / max_entropy,
                "peak_position": peak_pos,
                "head_type": head_type,
            })

    del model, tokenizer, attentions
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "experiment3_head_classification.csv")
    df.to_csv(path, index=False)

    print(f"\nSaved: {path}")
    print("\n=== Head type distribution ===")
    print(df["head_type"].value_counts().to_string())
    print(f"\n=== By layer ===")
    pivot = df.groupby(["layer", "head_type"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    # Plot 1: head type counts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts = df["head_type"].value_counts()
    colors = {
        "strong_sink": "#E74C3C",
        "moderate_sink": "#E67E22",
        "distributed": "#2ECC71",
        "recency": "#4C9BE8",
        "mixed": "#95A5A6",
    }
    bar_colors = [colors.get(t, "#95A5A6") for t in counts.index]
    axes[0].bar(counts.index, counts.values, color=bar_colors)
    axes[0].set_title("Head type distribution")
    axes[0].set_ylabel("Number of heads")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: heatmap of sink_share_first_4 by layer x head
    pivot2 = df.pivot(index="layer", columns="head", values="sink_share_first_4")
    im = axes[1].imshow(pivot2.values, aspect="auto", origin="lower", cmap="YlOrRd")
    axes[1].set_title("Sink share (first 4 tokens) by layer × head")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[1], label="Attention mass to first 4 tokens")

    plt.suptitle(f"{model_name} seq={seq_len} — Per-head attention sink classification",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp3_head_classification.png"), dpi=180,
                bbox_inches="tight")
    plt.close()
    print(f"Plot: plots/exp3_head_classification.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df1 = experiment_1_input_types(model_name="gpt2", seq_lens=[128, 256, 512])
    df2 = experiment_2_kv_ablation(model_name="gpt2", seq_len=256)
    df3 = experiment_3_head_classification(model_name="gpt2", seq_len=512)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS DONE")
    print("=" * 60)
    print(f"\nResults in: {RESULTS_DIR}/")
    print(f"Plots in:   {PLOTS_DIR}/")
