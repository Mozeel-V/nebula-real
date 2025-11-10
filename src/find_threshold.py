#!/usr/bin/env python3
"""
Search for (threshold, min_windows_required) combos that yield fp==0 and maximize tp,
given a window_eval.json (which contains per-sample 'scores' arrays).

Usage:
  python find_threshold.py \
    --we results/window_eval/window_eval.json \
    --grid 200 \
    --max_m 5 \
    --show_top 20 \
    --out results/best_combo_norm.json
Outputs printed combos (thr, m, tp, fp, tn, fn) and writes best-found JSON to best_combo.json
"""
'''
import json, argparse, numpy as np
from pathlib import Path

def load_we(p):
    return json.load(open(p, "r", encoding="utf-8"))

def pred_sample_by_minwins(scores, thr, minwins):
    # scores: list of floats per window
    if not scores:
        return 0
    wins = sum(1 for s in scores if s >= thr)
    return 1 if wins >= minwins else 0

def compute_counts(samples, thr, minwins):
    tp = fp = tn = fn = 0
    for sid, info in samples.items():
        lab = int(info.get("label", 0))
        scores = info.get("scores", [])
        pred = pred_sample_by_minwins(scores, thr, minwins)
        if lab == 1 and pred == 1:
            tp += 1
        elif lab == 1 and pred == 0:
            fn += 1
        elif lab == 0 and pred == 1:
            fp += 1
        elif lab == 0 and pred == 0:
            tn += 1
    return tp, fp, tn, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--we", required=True)
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--max_m", type=int, default=5, help="search min-windows required up to this value")
    ap.add_argument("--show_top", type=int, default=20)
    ap.add_argument("--out", default="results/best_combo.json")
    args = ap.parse_args()

    we = load_we(args.we)
    samples = we.get("samples", {})
    if not samples:
        print("No samples in", args.we); return

    # collect per-sample maxes if needed
    all_scores = []
    for sid, info in samples.items():
        all_scores.extend(info.get("scores", []))
    if not all_scores:
        print("No window scores found."); return

    # candidate thresholds: either unique scores or linspace
    uniq = sorted(set(all_scores))
    if len(uniq) <= args.grid:
        candidates = uniq
    else:
        candidates = list(np.linspace(0.0, 1.0, args.grid))

    results = []
    for thr in candidates:
        for m in range(1, args.max_m+1):
            tp, fp, tn, fn = compute_counts(samples, thr, m)
            results.append((fp, -tp, thr, m, tp, fp, tn, fn))
    # sort by fp asc then tp desc
    results.sort()
    # present top combos
    print("Top candidate (fp asc, tp desc):")
    for r in results[:args.show_top]:
        _, _neg_tp, thr, m, tp, fp, tn, fn = r
        print(f"thr={thr:.6f}  minwins={m}  tp={tp} fp={fp} tn={tn} fn={fn}")

    # find best with fp==0
    zero_fp = [r for r in results if r[5] == 0]  # r[5] is fp
    if zero_fp:
        zero_fp.sort(key=lambda x: (-x[4], x[2], x[3]))  # max tp, lower thr, lower m
        best = zero_fp[0]
        _, _, thr, m, tp, fp, tn, fn = best
        print("\nBest zero-FP combo found:")
        print(f"thr={thr:.6f}  minwins={m}  tp={tp} fp={fp} tn={tn} fn={fn}")
        out = {"thr": float(thr), "minwins": int(m), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("Wrote best combo to", args.out)
    else:
        print("\nNo (thr,m) found with fp==0 in grid. Showing top few minimizing fp:")
        for r in results[:args.show_top]:
            _, _neg_tp, thr, m, tp, fp, tn, fn = r
            print(f"thr={thr:.6f}  minwins={m}  tp={tp} fp={fp} tn={tn} fn={fn}")

if __name__ == "__main__":
    main()
'''

#!/usr/bin/env python3
"""
Find threshold + minwins combos that yield FP==0 and maximize TP.

Usage:
  python3 scripts/find_zero_fp_combo.py --we results/window_eval_before_vocabn/window_eval.json --max_minwins 6 --grid 400 --show 10

Saves best combos to results/best_combo_grid.json and prints them.
"""
import argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict

def load_window_eval(path):
    j = json.load(open(path, "r", encoding="utf-8"))
    samples = j.get("samples", {})
    return samples

def per_sample_maxes(samples):
    per = {}
    for sid, info in samples.items():
        scores = info.get("scores", [])
        per[sid] = {
            "label": int(info.get("label", 0)),
            "scores": [float(x) for x in scores],
            "n_windows": len(scores),
            "max": float(max(scores)) if scores else 0.0
        }
    return per

def search_grid(per_samples, num_thr=400, max_minwins=6, allow_fp_limit=None):
    # collect all window scores
    all_scores = []
    for sid, info in per_samples.items():
        all_scores.extend(info["scores"])
    all_scores = np.array(all_scores)
    if all_scores.size == 0:
        raise RuntimeError("No window scores found.")
    # candidate thresholds: unique sorted + small epsilon grid
    cand = np.unique(np.round(all_scores, 6))
    # if too many unique values, create linspace between min and max
    if cand.size > num_thr:
        cand = np.linspace(float(all_scores.min()), float(all_scores.max()), num=num_thr)
    # search
    results = []
    for thr in cand:
        for m in range(1, max_minwins+1):
            tp = fp = tn = fn = 0
            for sid, info in per_samples.items():
                lab = info["label"]
                # count windows >= thr
                cnt = sum(1 for s in info["scores"] if s >= thr)
                pred = 1 if cnt >= m else 0
                if lab==1 and pred==1: tp += 1
                if lab==1 and pred==0: fn += 1
                if lab==0 and pred==1: fp += 1
                if lab==0 and pred==0: tn += 1
            # optionally skip if fp > allow_fp_limit (if provided)
            if allow_fp_limit is not None and fp > allow_fp_limit:
                continue
            results.append((fp, -tp, thr, m, tp, fp, tn, fn))
    # sort ascending fp then descending tp
    results_sorted = sorted(results, key=lambda x: (x[0], x[1], -x[2]))
    return results_sorted

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--we", required=True, help="window_eval.json")
    p.add_argument("--max_minwins", type=int, default=6)
    p.add_argument("--grid", type=int, default=400)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--allow_fp_limit", type=int, default=0,
                   help="if set, allow FP up to this value when searching")
    p.add_argument("--out", default="results/best_combo_grid.json")
    args = p.parse_args()

    samples = load_window_eval(args.we)
    per = per_sample_maxes(samples)
    results = search_grid(per, num_thr=args.grid, max_minwins=args.max_minwins, allow_fp_limit=args.allow_fp_limit)

    if not results:
        print("No results found.")
        return

    # print top candidates (best: fp asc, tp desc)
    print("Top candidates (fp asc, tp desc):")
    for i, r in enumerate(results[:args.show]):
        fp, ntp_neg, thr, m, tp, fp, tn, fn = r
        print(f"{i+1:02d}) thr={thr:.6f} minwins={m} tp={tp} fp={fp} tn={tn} fn={fn}")

    # write best combo (first result)
    best = results[0]
    fp, ntp_neg, thr, m, tp, fp, tn, fn = best
    out = {"thr": float(thr), "minwins": int(m), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"grid_results_top": [ {"thr":float(r[2]), "minwins":int(r[3]), "tp":int(r[4]), "fp":int(r[5]), "tn":int(r[6]), "fn":int(r[7])} for r in results[:100]] , "best": out}, f, indent=2)
    print("Wrote best combos to", args.out)

if __name__ == "__main__":
    main()
