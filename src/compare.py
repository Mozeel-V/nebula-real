#!/usr/bin/env python3
"""
Compare BEFORE and AFTER attack evaluation results.

Inputs:
  --before : window_eval.json from clean dataset (baseline)
  --after  : window_eval.json from attacked dataset
  --zero_fp: JSON with threshold info (optional, from find_threshold_and_minwins.py)
  --out_dir: output folder for comparison plots and summary CSV

Outputs:
  - comparison_summary.csv: per-sample detection summary
  - per_sample/*.png: timeline plots showing P(malware) before vs after
  - global summary printed to stdout

Usage example:
  python compare.py \
    --before results/window_eval/window_eval.json \
    --after results/window_eval_after/window_eval.json \
    --zero_fp results/best_combo.json \
    --out_dir results/comparisons
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_window_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", {})

def compute_sample_maxes(samples):
    """Return {sid: max_score, label} for a given window_eval."""
    out = {}
    for sid, info in samples.items():
        scores = info.get("scores", [])
        if not scores:
            maxs = 0.0
        else:
            maxs = float(max(scores))
        out[sid] = {"label": int(info.get("label", 0)), "max": maxs}
    return out

def load_threshold(zero_fp_path):
    try:
        thr = 0.5
        minwins = 1
        if zero_fp_path and Path(zero_fp_path).exists():
            j = json.load(open(zero_fp_path, "r", encoding="utf-8"))
            thr = float(j.get("thr", thr))
            minwins = int(j.get("minwins", 1))
        return thr, minwins
    except Exception as e:
        print("[WARN] could not load threshold file:", e)
        return 0.5, 1

def plot_comparison(before_scores, after_scores, sid, lab, out_dir):
    """Plot before/after per-sample window scores."""
    fig, ax = plt.subplots(figsize=(6,2.4))
    ax.plot(before_scores, label="Before", color="C0", linewidth=0.6)
    ax.plot(after_scores, label="After", color="C3", linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("P(malware)")
    ax.set_title(f"SID={sid} label={lab}")
    ax.legend()
    outp = Path(out_dir) / f"{sid}.png"
    fig.tight_layout()
    fig.savefig(outp, dpi=120)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    #ap.add_argument("--zero_fp", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only_malware", action="store_true", help="compare only malware samples")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    per_sample_dir = outdir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    before = load_window_eval(args.before)
    after = load_window_eval(args.after)
    #thr, minwins = load_threshold(args.zero_fp)
    #print(f"[INFO] Using threshold={thr:.6f}, minwins={minwins}")
    thr = 0.848586

    before_max = compute_sample_maxes(before)
    after_max = compute_sample_maxes(after)

    sids = sorted(set(before_max.keys()) & set(after_max.keys()))
    print(f"[INFO] Found {len(sids)} common samples between before/after")

    # CSV setup
    summary_path = outdir / "comparison_summary.csv"
    csv_f = open(summary_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["sid", "label", "max_before", "max_after", "pred_before", "pred_after", "changed"])

    tp = tn = fp = fn = evaded = 0
    total = len(sids)
    for sid in sids:
        lab = before_max[sid]["label"]
        mb = before_max[sid]["max"]
        ma = after_max[sid]["max"]

        # classify by simple threshold
        pb = 1 if mb >= thr else 0
        pa = 1 if ma >= thr else 0

        # changed detection
        changed = pb != pa

        # count confusion
        if lab == 1:
            if pb == 1 and pa == 0:
                evaded += 1
            if pa == 1 and pb == 1:
                tp += 1
            elif pa == 0 and pb == 0:
                fn += 1
        elif lab == 0:
            if pa == 1 and pb == 1:
                fp += 1
            elif pa == 0 and pb == 0:
                tn += 1

        writer.writerow([sid, lab, mb, ma, pb, pa, int(changed)])

        if not args.only_malware or lab == 1:
            before_scores = before[sid].get("scores", [])
            after_scores = after[sid].get("scores", [])
            plot_comparison(before_scores, after_scores, sid, lab, per_sample_dir)

    csv_f.close()

    print(f"[DONE] Wrote {summary_path}")
    print(f"[INFO] Total={total}, TP={tp}, TN={tn}, FP={fp}, FN={fn}, Evaded={evaded}")
    print(f"[INFO] Per-sample plots in {per_sample_dir}")

if __name__ == "__main__":
    main()
