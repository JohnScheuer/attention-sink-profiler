import os
import re

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "results"
PLOTS_DIR = "plots"


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_sink_share_by_layer(summary):
    grouped = (
        summary.groupby(["model", "seq_len", "layer"], as_index=False)
        [["sink_share_first_4", "sink_share_first_8", "boost_first_4", "boost_first_8"]]
        .mean()
    )

    for model in grouped["model"].unique():
        sub = grouped[grouped["model"] == model].copy()

        plt.figure(figsize=(10, 6))
        for seq_len in sorted(sub["seq_len"].unique()):
            s = sub[sub["seq_len"] == seq_len].sort_values("layer")
            plt.plot(s["layer"], s["sink_share_first_4"], marker="o", label=f"seq={seq_len}")
        plt.title(f"{model} — attention sink share to first 4 tokens")
        plt.xlabel("Layer")
        plt.ylabel("Mean attention share to first 4 tokens")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{safe_name(model)}_sink_share_first4_by_layer.png"), dpi=180)
        plt.close()

        plt.figure(figsize=(10, 6))
        for seq_len in sorted(sub["seq_len"].unique()):
            s = sub[sub["seq_len"] == seq_len].sort_values("layer")
            plt.plot(s["layer"], s["boost_first_4"], marker="o", label=f"seq={seq_len}")
        plt.title(f"{model} — sink boost over uniform (first 4 tokens)")
        plt.xlabel("Layer")
        plt.ylabel("Boost over uniform")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{safe_name(model)}_boost_first4_by_layer.png"), dpi=180)
        plt.close()


def plot_heatmaps(positions):
    for model in positions["model"].unique():
        model_df = positions[positions["model"] == model]
        for seq_len in sorted(model_df["seq_len"].unique()):
            sub = model_df[model_df["seq_len"] == seq_len].copy()
            pivot = sub.pivot(index="layer", columns="key_pos", values="attn_mass").sort_index()

            plt.figure(figsize=(12, 6))
            plt.imshow(pivot.values, aspect="auto", origin="lower")
            plt.colorbar(label="Mean attention mass received")
            plt.title(f"{model} — seq_len={seq_len} — attention received by key position")
            plt.xlabel("Key position")
            plt.ylabel("Layer")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{safe_name(model)}_heatmap_seq{seq_len}.png"), dpi=180)
            plt.close()


def plot_peak_key_position(summary):
    grouped = (
        summary.groupby(["model", "seq_len", "layer"], as_index=False)["peak_key_pos"]
        .mean()
    )

    for model in grouped["model"].unique():
        sub = grouped[grouped["model"] == model]
        plt.figure(figsize=(10, 6))
        for seq_len in sorted(sub["seq_len"].unique()):
            s = sub[sub["seq_len"] == seq_len].sort_values("layer")
            plt.plot(s["layer"], s["peak_key_pos"], marker="o", label=f"seq={seq_len}")
        plt.title(f"{model} — mean peak-attended key position by layer")
        plt.xlabel("Layer")
        plt.ylabel("Mean peak key position")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{safe_name(model)}_peak_keypos_by_layer.png"), dpi=180)
        plt.close()


def main():
    ensure_dirs()

    summary = pd.read_csv(os.path.join(RESULTS_DIR, "attention_sink_summary.csv"))
    positions = pd.read_csv(os.path.join(RESULTS_DIR, "attention_sink_positions.csv"))

    plot_sink_share_by_layer(summary)
    plot_heatmaps(positions)
    plot_peak_key_position(summary)

    print("Plots written to plots/")


if __name__ == "__main__":
    main()
