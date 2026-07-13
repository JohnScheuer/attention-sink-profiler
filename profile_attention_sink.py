import argparse
import gc
import json
import math
import os
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_TEXT = (
    "In a long technical discussion, researchers analyze language models, "
    "attention patterns, memory behavior, token generation, and inference efficiency. "
    "They compare latency, throughput, cache policies, and routing effects across "
    "different workloads while carefully measuring the contribution of each token "
    "position to the final output distribution. "
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "gpt2-medium"],
        help="HF model names",
    )
    p.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Sequence lengths to profile",
    )
    p.add_argument(
        "--tail-window",
        type=int,
        default=64,
        help="How many final query positions to average over",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Output directory",
    )
    return p.parse_args()


def ensure_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def build_input_ids(tokenizer, seq_len):
    prefix = (tokenizer.eos_token + " ") if tokenizer.eos_token else ""
    chunk_text = prefix + BASE_TEXT
    chunk_ids = tokenizer(
        chunk_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    if chunk_ids.numel() == 0:
        raise RuntimeError("Tokenizer produced zero tokens")

    parts = []
    total = 0
    while total < seq_len:
        parts.append(chunk_ids)
        total += chunk_ids.numel()

    ids = torch.cat(parts, dim=0)[:seq_len]
    return ids.unsqueeze(0)


def expected_uniform_share(seq_len, q_start, q_end, k):
    vals = []
    for q in range(q_start, q_end):
        valid = q + 1
        vals.append(min(k, valid) / valid)
    return float(sum(vals) / len(vals))


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

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
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    model.config.output_attentions = True
    model.eval()
    model.to(device)

    return tokenizer, model


def analyze_one_run(model_name, seq_len, tail_window, attentions):
    # attentions: tuple[layer] of [batch, heads, q, k]
    rows_summary = []
    rows_position = []

    num_layers = len(attentions)
    q_len = attentions[0].shape[-2]
    k_len = attentions[0].shape[-1]
    assert q_len == seq_len and k_len == seq_len

    tail = min(tail_window, seq_len)
    q_start = seq_len - tail
    q_end = seq_len

    uniform_1 = expected_uniform_share(seq_len, q_start, q_end, 1)
    uniform_4 = expected_uniform_share(seq_len, q_start, q_end, 4)
    uniform_8 = expected_uniform_share(seq_len, q_start, q_end, 8)

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: [1, heads, q, k]
        attn = layer_attn[0].float().cpu()  # [heads, q, k]
        heads = attn.shape[0]

        tail_attn = attn[:, q_start:q_end, :seq_len]   # [heads, tail, k]
        mean_received_by_head = tail_attn.mean(dim=1)  # [heads, k]
        layer_mean_received = mean_received_by_head.mean(dim=0)  # [k]

        for key_pos in range(seq_len):
            rows_position.append({
                "model": model_name,
                "seq_len": seq_len,
                "layer": layer_idx,
                "key_pos": key_pos,
                "attn_mass": float(layer_mean_received[key_pos].item()),
                "tail_window": tail,
            })

        argmax_positions = tail_attn.argmax(dim=-1)  # [heads, tail]

        for head_idx in range(heads):
            dist = mean_received_by_head[head_idx]

            sink_1 = float(dist[:1].sum().item())
            sink_4 = float(dist[:min(4, seq_len)].sum().item())
            sink_8 = float(dist[:min(8, seq_len)].sum().item())

            peak_key_pos = int(dist.argmax().item())
            peak_key_mass = float(dist.max().item())

            top1_first4_frac = float((argmax_positions[head_idx] < min(4, seq_len)).float().mean().item())
            top1_first8_frac = float((argmax_positions[head_idx] < min(8, seq_len)).float().mean().item())

            rows_summary.append({
                "model": model_name,
                "seq_len": seq_len,
                "layer": layer_idx,
                "head": head_idx,
                "tail_window": tail,
                "sink_share_first_1": sink_1,
                "sink_share_first_4": sink_4,
                "sink_share_first_8": sink_8,
                "uniform_first_1": uniform_1,
                "uniform_first_4": uniform_4,
                "uniform_first_8": uniform_8,
                "boost_first_1": sink_1 / uniform_1 if uniform_1 > 0 else None,
                "boost_first_4": sink_4 / uniform_4 if uniform_4 > 0 else None,
                "boost_first_8": sink_8 / uniform_8 if uniform_8 > 0 else None,
                "peak_key_pos": peak_key_pos,
                "peak_key_mass": peak_key_mass,
                "top1_first4_query_frac": top1_first4_frac,
                "top1_first8_query_frac": top1_first8_frac,
            })

    return rows_summary, rows_position


def main():
    args = parse_args()
    ensure_dirs(args.outdir)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": args.models,
        "seq_lens": args.seq_lens,
        "tail_window": args.tail_window,
    }

    all_summary = []
    all_positions = []
    run_log = []

    print(json.dumps(metadata, indent=2))

    for model_name in args.models:
        print(f"\n=== Loading {model_name} ===")
        tokenizer, model = load_model_and_tokenizer(model_name, device)
        max_positions = getattr(model.config, "n_positions", 1024)
        print(f"max_positions={max_positions}")

        for seq_len in args.seq_lens:
            if seq_len > max_positions:
                print(f"[skip] {model_name} seq_len={seq_len} > max_positions={max_positions}")
                continue

            print(f"[run] model={model_name} seq_len={seq_len}")
            t0 = time.time()

            try:
                input_ids = build_input_ids(tokenizer, seq_len).to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        output_attentions=True,
                        use_cache=False,
                        return_dict=True,
                    )

                attentions = getattr(outputs, "attentions", None)
                if attentions is None or len(attentions) == 0:
                    raise RuntimeError("Model returned no attentions; eager attention backend may be required")

                attentions_cpu = [a.detach().cpu() for a in attentions if a is not None]
                if len(attentions_cpu) == 0:
                    raise RuntimeError("Attention list was empty after filtering None values")
                del outputs
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

                rows_summary, rows_position = analyze_one_run(
                    model_name=model_name,
                    seq_len=seq_len,
                    tail_window=args.tail_window,
                    attentions=attentions_cpu,
                )

                all_summary.extend(rows_summary)
                all_positions.extend(rows_position)

                dt = time.time() - t0
                print(f"[ok] model={model_name} seq_len={seq_len} rows_summary={len(rows_summary)} rows_position={len(rows_position)} time={dt:.2f}s")

                run_log.append({
                    "model": model_name,
                    "seq_len": seq_len,
                    "status": "ok",
                    "seconds": dt,
                })

                del attentions_cpu
                del input_ids
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                dt = time.time() - t0
                print(f"[fail] model={model_name} seq_len={seq_len} error={e}")
                run_log.append({
                    "model": model_name,
                    "seq_len": seq_len,
                    "status": "fail",
                    "seconds": dt,
                    "error": str(e),
                })
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

        del model
        del tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summary)
    positions_df = pd.DataFrame(all_positions)
    runlog_df = pd.DataFrame(run_log)

    summary_path = os.path.join(args.outdir, "attention_sink_summary.csv")
    positions_path = os.path.join(args.outdir, "attention_sink_positions.csv")
    runlog_path = os.path.join(args.outdir, "run_log.csv")
    metadata_path = os.path.join(args.outdir, "metadata.json")

    summary_df.to_csv(summary_path, index=False)
    positions_df.to_csv(positions_path, index=False)
    runlog_df.to_csv(runlog_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Done ===")
    print(f"summary:   {summary_path}")
    print(f"positions: {positions_path}")
    print(f"run_log:   {runlog_path}")
    print(f"metadata:  {metadata_path}")


if __name__ == "__main__":
    main()
