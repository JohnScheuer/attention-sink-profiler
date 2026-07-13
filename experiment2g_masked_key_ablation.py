import gc
import math
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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
            model_name,
            dtype=dtype,
            attn_implementation="eager",
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )

    model.eval().to(DEVICE)
    return tokenizer, model


def build_ids(tokenizer, seq_len):
    chunk = tokenizer(
        NATURAL_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    parts = []
    total = 0
    while total < seq_len:
        parts.append(chunk)
        total += chunk.numel()

    ids = torch.cat(parts)[:seq_len]
    return ids.unsqueeze(0).to(DEVICE)


def make_attention_mask(seq_len, masked_spans):
    """
    GPT-2 attention_mask: 1 = keep, 0 = masked key.
    This masks selected key positions for all queries.
    """
    mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)
    for start, end in masked_spans:
        start = max(0, start)
        end = min(seq_len, end)
        if start < end:
            mask[:, start:end] = 0
    return mask


def run_tail_metrics(model, input_ids, attention_mask=None, tail_window=64):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = out.logits.float()  # [1, seq, vocab]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, -1)  # [1, seq-1]

    tail = min(tail_window, token_losses.shape[1])
    tail_losses = token_losses[:, -tail:]

    mean_tail_loss = tail_losses.mean().item()
    tail_ppl = math.exp(mean_tail_loss)

    last_logits = logits[0, -1, :].cpu()
    last_probs = torch.softmax(last_logits, dim=0)

    return {
        "tail_loss": mean_tail_loss,
        "tail_ppl": tail_ppl,
        "last_logits": last_logits,
        "last_probs": last_probs,
    }


def kl_divergence(p, q, eps=1e-10):
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps))).item()


def topk_overlap(logits_a, logits_b, k=5):
    a = set(logits_a.topk(k).indices.tolist())
    b = set(logits_b.topk(k).indices.tolist())
    return len(a & b) / float(k)


def build_mask_specs(seq_len, tail_window=64, seed=123):
    specs = []

    # Sink masks
    for k in [1, 4, 8]:
        specs.append({
            "mask_name": f"first_{k}",
            "masked_spans": [(0, k)],
            "mask_size": k,
            "mask_group": "sink",
        })

    # Middle masks
    for k in [4, 8]:
        start = max(0, (seq_len // 2) - (k // 2))
        specs.append({
            "mask_name": f"middle_{k}",
            "masked_spans": [(start, start + k)],
            "mask_size": k,
            "mask_group": "middle",
        })

    # Recent masks: just before the evaluated tail window
    for k in [4, 8]:
        start = max(0, seq_len - tail_window - k)
        specs.append({
            "mask_name": f"recent_{k}",
            "masked_spans": [(start, start + k)],
            "mask_size": k,
            "mask_group": "recent",
        })

    # Random masks
    rng = random.Random(seed + seq_len)
    for k in [4, 8]:
        max_start = max(1, seq_len - tail_window - k - 1)
        for trial in range(3):
            start = rng.randint(1, max_start)
            specs.append({
                "mask_name": f"random_{k}_trial{trial}",
                "masked_spans": [(start, start + k)],
                "mask_size": k,
                "mask_group": f"random_{k}",
            })

    return specs


def main():
    print("=" * 60)
    print("EXPERIMENT 2g: Masked-key ablation")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    tokenizer, model = load_model("gpt2")
    rows = []

    for seq_len in [256, 512, 1024]:
        tail_window = 64
        print(f"\n--- seq_len={seq_len} ---")

        input_ids = build_ids(tokenizer, seq_len)

        baseline_mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)
        baseline = run_tail_metrics(
            model,
            input_ids,
            attention_mask=baseline_mask,
            tail_window=tail_window,
        )

        baseline_top1 = int(baseline["last_logits"].argmax().item())

        print(
            f"baseline tail_ppl={baseline['tail_ppl']:.4f}  "
            f"tail_loss={baseline['tail_loss']:.4f}"
        )

        rows.append({
            "seq_len": seq_len,
            "mask_name": "baseline",
            "mask_group": "baseline",
            "mask_size": 0,
            "tail_loss": baseline["tail_loss"],
            "tail_ppl": baseline["tail_ppl"],
            "ppl_ratio": 1.0,
            "loss_delta": 0.0,
            "last_token_kl": 0.0,
            "top1_changed": 0,
            "top5_overlap": 1.0,
        })

        for spec in build_mask_specs(seq_len, tail_window=tail_window):
            attn_mask = make_attention_mask(seq_len, spec["masked_spans"])
            result = run_tail_metrics(
                model,
                input_ids,
                attention_mask=attn_mask,
                tail_window=tail_window,
            )

            last_kl = kl_divergence(baseline["last_probs"], result["last_probs"])
            top1_changed = int(result["last_logits"].argmax().item() != baseline_top1)
            top5 = topk_overlap(baseline["last_logits"], result["last_logits"], k=5)

            row = {
                "seq_len": seq_len,
                "mask_name": spec["mask_name"],
                "mask_group": spec["mask_group"],
                "mask_size": spec["mask_size"],
                "tail_loss": result["tail_loss"],
                "tail_ppl": result["tail_ppl"],
                "ppl_ratio": result["tail_ppl"] / baseline["tail_ppl"],
                "loss_delta": result["tail_loss"] - baseline["tail_loss"],
                "last_token_kl": last_kl,
                "top1_changed": top1_changed,
                "top5_overlap": top5,
            }
            rows.append(row)

            print(
                f"{spec['mask_name']:16s}  "
                f"ppl_ratio={row['ppl_ratio']:.4f}  "
                f"loss_delta={row['loss_delta']:.4f}  "
                f"KL={row['last_token_kl']:.5f}  "
                f"top5={row['top5_overlap']:.1%}"
            )

        gc.collect()
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "experiment2g_masked_key_ablation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Aggregate random trials
    agg_rows = []
    for seq_len in sorted(df["seq_len"].unique()):
        sub = df[(df["seq_len"] == seq_len) & (df["mask_name"] != "baseline")].copy()

        fixed = sub[~sub["mask_group"].str.startswith("random_")].copy()
        agg_rows.append(fixed)

        for group in ["random_4", "random_8"]:
            g = sub[sub["mask_group"] == group]
            if len(g) > 0:
                agg_rows.append(pd.DataFrame([{
                    "seq_len": seq_len,
                    "mask_name": f"{group}_mean",
                    "mask_group": group,
                    "mask_size": int(g["mask_size"].iloc[0]),
                    "tail_loss": g["tail_loss"].mean(),
                    "tail_ppl": g["tail_ppl"].mean(),
                    "ppl_ratio": g["ppl_ratio"].mean(),
                    "loss_delta": g["loss_delta"].mean(),
                    "last_token_kl": g["last_token_kl"].mean(),
                    "top1_changed": g["top1_changed"].mean(),
                    "top5_overlap": g["top5_overlap"].mean(),
                }]))

    plot_df = pd.concat(agg_rows, ignore_index=True)
    plot_csv = os.path.join(RESULTS_DIR, "experiment2g_masked_key_ablation_agg.csv")
    plot_df.to_csv(plot_csv, index=False)
    print(f"Saved: {plot_csv}")

    # Plot 1: ppl_ratio by mask
    for metric in ["ppl_ratio", "loss_delta", "last_token_kl"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, seq_len in zip(axes, [256, 512, 1024]):
            sub = plot_df[plot_df["seq_len"] == seq_len].copy()
            sub = sub.sort_values(metric, ascending=False)

            ax.bar(sub["mask_name"], sub[metric])
            ax.set_title(f"seq_len={seq_len}")
            ax.set_xlabel("Mask")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=60)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Masked-key ablation — {metric}", y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"exp2g_{metric}.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    # Plot 2: grouped comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    groups_in_order = [
        "first_1", "first_4", "first_8",
        "middle_4", "middle_8",
        "recent_4", "recent_8",
        "random_4_mean", "random_8_mean",
    ]
    for ax, seq_len in zip(axes, [256, 512, 1024]):
        sub = plot_df[plot_df["seq_len"] == seq_len].copy()
        sub = sub.set_index("mask_name").reindex(groups_in_order).reset_index()
        ax.bar(sub["mask_name"], sub["ppl_ratio"])
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_title(f"seq_len={seq_len}")
        ax.set_ylabel("Tail PPL ratio")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Masked-key ablation — sink vs middle vs recent vs random", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "exp2g_grouped_ppl_ratio.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # Terminal summary
    print("\n=== Summary: strongest masks by ppl_ratio ===")
    print(
        plot_df.sort_values(["seq_len", "ppl_ratio"], ascending=[True, False])[
            ["seq_len", "mask_name", "ppl_ratio", "loss_delta", "last_token_kl", "top5_overlap"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
