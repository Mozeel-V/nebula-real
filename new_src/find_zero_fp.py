#!/usr/bin/env python3
"""
Find zero-false-positive thresholds (and variants) from a window_eval.json file.

Writes:
 - <out_dir>/best_combo.json         (best thr/minwins/consec chosen)
 - <out_dir>/threshold_candidates.csv (top candidates sorted by fp asc, tp desc)
 - <out_dir>/full_grid.csv           (complete grid evaluated)
 - <out_dir>/score_hist.png
 - <out_dir>/summary.json

Usage:
 python new_src/find_zero_fp.py \
   --we new_results/window_eval_after/window_eval.json \
   --out_dir new_results/zfp_after_retrain \
   --grid_samples 1000 \
   --max_minwins 5 \
   --max_consec 4
"""
import argparse, json, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_we(path):
    j = json.load(open(path, "r", encoding="utf-8"))
    if "samples" in j:
        samples = j["samples"]
    else:
        # maybe the file itself is a list / different layout
        samples = j
    # ensure samples keys are strings and values have 'label' and 'scores'
    parsed = {}
    for sid, info in samples.items():
        parsed[str(sid)] = {
            "label": int(info.get("label", 0)),
            "scores": [float(x) for x in info.get("scores", [])]
        }
    return parsed

def sample_pred_from_scores(scores, thr, minwins=0, consecutive_k=0):
    cls = [1 if s >= thr else 0 for s in scores]
    if consecutive_k and consecutive_k > 0:
        run = 0
        for v in cls:
            if v == 1:
                run += 1
                if run >= consecutive_k:
                    return 1
            else:
                run = 0
        return 0
    if minwins and minwins > 0:
        return 1 if sum(cls) >= minwins else 0
    return 1 if any(cls) else 0

def evaluate_thresholds(samples, thresholds, max_minwins=1, max_consec=0):
    """
    Evaluate thresholds over samples.
    returns list of dict rows with keys:
     thr, minwins, consecutive_k, tp,fp,tn,fn, tp_rate, fp_rate
    """
    rows = []
    # precompute per-sample max or scores list
    for thr in thresholds:
        for minwins in range(0, max_minwins+1):
            for consec in range(0, max_consec+1):
                tp = fp = tn = fn = 0
                for sid, info in samples.items():
                    y = info["label"]
                    pred = sample_pred_from_scores(info["scores"], thr, minwins=minwins, consecutive_k=consec)
                    if y == 1 and pred == 1:
                        tp += 1
                    elif y == 1 and pred == 0:
                        fn += 1
                    elif y == 0 and pred == 1:
                        fp += 1
                    elif y == 0 and pred == 0:
                        tn += 1
                rows.append({
                    "thr": float(thr),
                    "minwins": int(minwins),
                    "consecutive_k": int(consec),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "n_samples": tp+fp+tn+fn
                })
    return rows

def pick_best_zero_fp(rows):
    # filter rows with fp == 0
    z = [r for r in rows if r["fp"] == 0]
    if not z:
        return None
    # sort by tp desc, then tn desc then minwins asc then consecutive_k asc then thr asc
    z_sorted = sorted(z, key=lambda r: (-r["tp"], -r["tn"], r["minwins"], r["consecutive_k"], r["thr"]))
    return z_sorted[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--we", required=True, help="window_eval.json to analyze (new model)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid_samples", type=int, default=200, help="number of thresholds to try (use 0 to use all unique scores)")
    p.add_argument("--max_minwins", type=int, default=1, help="test minwins in [0..max_minwins]")
    p.add_argument("--max_consec", type=int, default=0, help="test consecutive_k in [0..max_consec]")
    p.add_argument("--initial_zero_fp", type=str, default="/mnt/data/zero_fp_threshold.json", help="optional existing zero_fp JSON (uploaded). Used for debug only.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_we(args.we)
    n_samples = len(samples)
    print(f"[info] Loaded {n_samples} samples from {args.we}")

    # collect all window scores
    all_scores = []
    per_sample_max = []
    for sid, info in samples.items():
        scs = info["scores"]
        if scs:
            all_scores.extend(scs)
            per_sample_max.append(max(scs))
    all_scores = np.array(all_scores) if all_scores else np.array([])
    print(f"[info] Total window scores: {len(all_scores)}")

    # thresholds to try
    if args.grid_samples <= 0:
        thresholds = sorted(set(all_scores))
    else:
        if len(all_scores) == 0:
            thresholds = [0.5]
        else:
            lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
            thresholds = np.linspace(lo, hi, args.grid_samples)
    thresholds = sorted(list(set([float(round(t, 12)) for t in thresholds])))
    print(f"[info] Trying {len(thresholds)} thresholds (grid_samples={args.grid_samples})")

    # diagnostics: write score stats and histogram
    stats = {}
    if len(all_scores) > 0:
        stats["min"] = float(np.min(all_scores)); stats["max"] = float(np.max(all_scores))
        stats["mean"] = float(np.mean(all_scores)); stats["median"] = float(np.median(all_scores))
        stats["std"] = float(np.std(all_scores))
    else:
        stats = {"min": None, "max": None, "mean": None}
    json.dump({"score_stats": stats, "n_samples": n_samples}, open(outdir/"score_stats.json","w"), indent=2)
    # histogram
    if len(all_scores) > 0:
        plt.figure(figsize=(6,3))
        plt.hist(all_scores, bins=200)
        plt.title("Window score histogram")
        plt.xlabel("P(malware)")
        plt.tight_layout()
        plt.savefig(outdir/"score_hist.png", dpi=150)
        plt.close()
        print(f"[info] Wrote score histogram to {outdir/'score_hist.png'}")

    # evaluate grid
    rows = evaluate_thresholds(samples, thresholds, max_minwins=args.max_minwins, max_consec=args.max_consec)
    print(f"[info] Evaluated grid: {len(rows)} entries")

    # write full grid CSV
    csv_fields = ["thr","minwins","consecutive_k","tp","fp","tn","fn","n_samples"]
    with open(outdir/"full_grid.csv","w",newline="",encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"[info] Wrote full grid to {outdir/'full_grid.csv'}")

    # pick best zero-fp
    best = pick_best_zero_fp(rows)
    if best is None:
        print("[warn] No threshold in grid achieves FP == 0. Showing top candidates by fp asc, tp desc")
        # list top 30 by (fp asc, -tp)
        sorted_rows = sorted(rows, key=lambda r: (r["fp"], -r["tp"], -r["tn"]))
        top = sorted_rows[:30]
    else:
        print(f"[info] Best zero-FP candidate: thr={best['thr']} minwins={best['minwins']} consec={best['consecutive_k']} tp={best['tp']} fp={best['fp']} tn={best['tn']} fn={best['fn']}")
        top = [best]

    # write candidates CSV (top 200 sorted by fp asc, tp desc)
    rows_sorted = sorted(rows, key=lambda r: (r["fp"], -r["tp"], -r["tn"]))
    with open(outdir/"threshold_candidates.csv","w",newline="",encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=csv_fields)
        w.writeheader()
        for r in rows_sorted[:1000]:
            w.writerow({k: r[k] for k in csv_fields})
    print(f"[info] Wrote threshold candidates to {outdir/'threshold_candidates.csv'}")

    # write best combo json (if found)
    best_combo = best if best is not None else rows_sorted[0]
    json.dump({"best_combo": best_combo, "top_candidates": rows_sorted[:30]}, open(outdir/"best_combo.json","w"), indent=2)
    print(f"[info] Wrote best combo to {outdir/'best_combo.json'}")

    # summary
    summary = {
        "n_samples": n_samples,
        "grid_samples": len(thresholds),
        "max_minwins": args.max_minwins,
        "max_consecutive_k": args.max_consec,
        "score_stats": stats,
        "best_combo": best_combo
    }
    json.dump(summary, open(outdir/"summary.json","w"), indent=2)
    print(f"[info] Wrote summary to {outdir/'summary.json'}")

    if args.debug:
        print("[debug] top candidates (fp asc, tp desc):")
        for r in rows_sorted[:30]:
            print(f" thr={r['thr']:.6f} minwins={r['minwins']} consec={r['consecutive_k']} tp={r['tp']} fp={r['fp']} tn={r['tn']} fn={r['fn']}")

if __name__ == "__main__":
    main()
