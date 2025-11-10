#!/usr/bin/env python3
"""
Window-level evaluation script.

Saves:
 - <out_dir>/window_eval.json  (per-sample window scores and metadata)
 - <out_dir>/window_eval.score_hist.png  (global histogram)
 - optionally <out_dir>/window_figures/per_sample/<sid>.png

Usage example:
Before
  python window_eval_plot.py \
    --data_file dataset_small_2k.tsv \
    --ckpt checkpoints/best.pt \
    --vocab checkpoints/vocab.json \
    --out_dir results/window_eval \
    --window_unit event \
    --events_per_window 16 \
    --stride_events 4 \
    --batch_size 128 \
    --save_plots

After
    python window_eval_plot.py \
    --data_file results/attacks/saliency_attack.tsv \
    --ckpt checkpoints/best.pt \
    --vocab checkpoints/vocab.json \
    --out_dir results/window_eval_after \
    --events_per_window 16 --stride_events 4 --batch_size 128 --save_plots

"""
import argparse
import json
import math
import os
from pathlib import Path
from time import time
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

# local imports (tokenizer + model)
from tokenizer import tokenize, load_vocab, tokens_to_ids
from nebula_model import NebulaTiny

# helper to load model with config in checkpoint
def load_model_from_checkpoint(ckpt_path, vocab_path=None, device=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config") or ck.get("cfg") or {}
    # fallback to reasonable defaults
    vocab = None
    if vocab_path:
        vocab = load_vocab(vocab_path)
    vocab_size = cfg.get("vocab_size", len(vocab) if vocab else 30000)
    d_model = cfg.get("d_model", 128)
    nhead = cfg.get("nhead", cfg.get("heads", 4))
    num_layers = cfg.get("num_layers", cfg.get("layers", 2))
    ff = cfg.get("ff", cfg.get("dim_feedforward", 256))
    max_len = cfg.get("max_len", 512)
    num_classes = cfg.get("num_classes", 2)
    chunk_size = cfg.get("chunk_size", 0)

    model = NebulaTiny(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff,
        max_len=max_len,
        num_classes=num_classes,
        chunk_size=chunk_size
    )

    # find state dict under common keys
    state = ck.get("model", ck.get("model_state", ck.get("state_dict", ck)))
    if state is None:
        raise RuntimeError(f"No model weights found in checkpoint {ckpt_path}; keys: {list(ck.keys())}")
    # load state dict with non-strict to be robust to small mismatches
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print("[WARN] Missing keys when loading state_dict:", missing.missing_keys)
    if missing.unexpected_keys:
        print("[WARN] Unexpected keys:", missing.unexpected_keys)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device).eval()
    return model, cfg

def generate_windows_from_trace(trace, window_unit, events_per_window, stride_events):
    if window_unit == "event":
        if " ||| " in trace:
            tokens = trace.split(" ||| ")
        else:
            tokens = trace.split()
        windows = []
        T = len(tokens)
        i = 0
        while i < T:
            win = tokens[i:i+events_per_window]
            if not win:
                break
            windows.append(" ||| ".join(win))
            i += stride_events
        return windows
    else:
        # fallback: token windows
        tokens = tokenize(trace)
        windows = []
        T = len(tokens)
        i = 0
        while i < T:
            win = tokens[i:i+events_per_window]
            if not win:
                break
            windows.append(" ".join(win))
            i += stride_events
        return windows

def batch_infer(model, device, vocab, windows, max_len, batch_size):
    """
    Tokenize list of windows and run batched inference.
    Returns list of probabilities (malware class prob).
    """
    all_ids = []
    for w in windows:
        toks = tokenize(w)
        ids = tokens_to_ids(toks, vocab, max_len=max_len)
        all_ids.append(ids)
    # pad batches and run
    probs = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i+batch_size]
        maxl = max(len(x) for x in batch)
        arr = np.zeros((len(batch), maxl), dtype=np.int64)
        for j, row in enumerate(batch):
            arr[j, :len(row)] = row
        tensor = torch.tensor(arr, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(tensor)  # [B, num_classes]
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        probs.extend(p)
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_unit", default="event")
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--sample_limit", type=int, default=0, help="0 => all samples")
    ap.add_argument("--save_plots", action="store_true", help="save per-sample timeline plots (disk-heavy)")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs_dir = outdir / "window_figures" / "per_sample_class"
    if args.save_plots:
        figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_checkpoint(args.ckpt, args.vocab, device=device)
    vocab = load_vocab(args.vocab)
    max_len = min(args.max_len, int(cfg.get("max_len", args.max_len)))

    print("Device:", device)
    print("Loaded model config:", cfg)
    print("Vocab size:", len(vocab))

    samples = []
    with open(args.data_file, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            sid, lab, trace = ln.split("\t", 2)
            samples.append((sid, int(lab), trace))
            if args.sample_limit and len(samples) >= args.sample_limit:
                break
    print("Loaded samples:", len(samples))

    results = {"meta": {"data_file": args.data_file, "ckpt": args.ckpt, "vocab": args.vocab, "cfg": cfg},
               "samples": {}}

    # iterate and score windows per sample
    global_scores = []
    t0 = time()
    for i, (sid, lab, trace) in enumerate(samples):
        windows = generate_windows_from_trace(trace, args.window_unit, args.events_per_window, args.stride_events)
        if len(windows) == 0:
            # treat empty as single window with empty score
            windows = [""]
        probs = batch_infer(model, device, vocab, windows, max_len=max_len, batch_size=args.batch_size)
        # ensure same length even if model produced fewer probs
        scores = [float(x) for x in probs]
        results["samples"][sid] = {
            "label": lab,
            "n_windows": len(windows),
            "scores": scores
        }
        global_scores.extend(scores)

        # quick per-sample plot (optional)
        if args.save_plots:
            try:
                fig, ax = plt.subplots(figsize=(6,2.4))
                ax.plot(range(len(scores)), scores, marker=".", linewidth=0.6)
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f"SID={sid} label={lab} nwin={len(scores)}")
                ax.set_xlabel("window_index")
                ax.set_ylabel("P(malware)")
                outp = figs_dir / f"{sid}.png"
                fig.tight_layout()
                fig.savefig(outp, dpi=120)
                plt.close(fig)
            except Exception as e:
                print("Plot error for sid", sid, e)

        if (i+1) % 100 == 0:
            print(f"Scored {i+1}/{len(samples)} samples")

    elapsed = time() - t0
    print(f"Scored all samples in {elapsed:.1f}s")

    # write window_eval.json
    out_json = outdir / "window_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print("Wrote", out_json)

    # summary hist and threshold candidates
    if len(global_scores) == 0:
        print("No scores produced.")
        return
    vals = np.array(global_scores)
    plt.figure(figsize=(6,3.5))
    plt.hist(vals, bins=200, alpha=0.8)
    plt.xlabel("P(malware)")
    plt.ylabel("count")
    plt.title("Window score histogram")
    hist_out = outdir / "window_eval.score_hist.png"
    plt.tight_layout()
    plt.savefig(hist_out, dpi=140)
    plt.close()
    print("Saved global histogram to", hist_out)

    # build candidate thresholds and compute simple summary table
    cand = np.unique(np.round(vals, 6))
    cand = np.clip(cand, 0.0, 1.0)
    # compute per-sample max to derive sample-level predictions quickly
    per_sample_max = {}
    for sid, info in results["samples"].items():
        scs = info.get("scores", [])
        per_sample_max[sid] = max(scs) if scs else 0.0
    # search a small set of thresholds (dense)
    grid = np.linspace(0.0, 1.0, num=200)
    summary = []
    for thr in grid:
        tp = fp = tn = fn = 0
        for sid, info in results["samples"].items():
            lab = int(info["label"])
            pred = 1 if per_sample_max[sid] >= thr else 0
            if lab == 1 and pred == 1:
                tp += 1
            elif lab == 1 and pred == 0:
                fn += 1
            elif lab == 0 and pred == 1:
                fp += 1
            elif lab == 0 and pred == 0:
                tn += 1
        summary.append({"thr": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    summary_path = outdir / "window_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary_grid": summary}, f, indent=2)
    print("Wrote threshold summary to", summary_path)

if __name__ == "__main__":
    main()
