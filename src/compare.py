#!/usr/bin/env python3
"""
Compare BEFORE and AFTER window evaluations and create **class-timeline** plots
(0 = benign window, 1 = malware window) side-by-side per sample.

- Loads window_eval JSONs (before/after) and a threshold JSON (accepts "thr" or "threshold")
- Converts per-window probabilities -> classes using thr
- Produces per-sample PNGs into out_dir/per_sample/ (filename sample_<sid>.png)
- Produces a CSV / JSON summary with counts (mal windows before/after, flipped windows)

Usage:
  n
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_threshold(z):
    # accept several common keys
    for k in ("thr", "threshold", "thr_val", "value"):
        if k in z:
            return float(z[k])
    # legacy: maybe saved as {"thr":..., "minwins":...}
    if "thr" in z:
        return float(z["thr"])
    raise KeyError("no threshold key found in zero_fp JSON (expected 'thr' or 'threshold')")

def scores_to_classes(scores, thr):
    # convert probabilities to 0/1 using threshold (>= thr -> 1)
    return [1 if (float(s) >= thr - 1e-12) else 0 for s in scores]

def plot_classes(before_cls, after_cls, sid, label, outpath, thr, show_prob_overlay=False, before_probs=None, after_probs=None):
    # produce clean side-by-side class timelines (step)
    nb = len(before_cls); na = len(after_cls)
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    xs_b = np.arange(nb); xs_a = np.arange(na)
    axs[0].step(xs_b, before_cls, where="mid", linewidth=1.6)
    axs[0].set_ylim(-0.12, 1.12)
    axs[0].set_yticks([0,1]); axs[0].set_yticklabels(["benign","malware"])
    axs[0].set_xlabel("window index"); axs[0].set_title(f"Before — sample {sid}  (label={label})")
    axs[1].step(xs_a, after_cls, where="mid", linewidth=1.6)
    axs[1].set_xlabel("window index"); axs[1].set_title(f"After — sample {sid}  (thr={thr:.6f})")

    # highlight successful evasions (1->0)
    for i in range(min(nb,na)):
        if before_cls[i] == 1 and after_cls[i] == 0:
            for ax in axs:
                ax.axvspan(i-0.3, i+0.3, color="#4caf50", alpha=0.25)

    # optionally overlay probabilities as faint line (for debugging only)
    if show_prob_overlay and before_probs is not None and after_probs is not None:
        ax0r = axs[0].twinx()
        ax0r.plot(xs_b, before_probs, linewidth=0.7, linestyle=":", alpha=0.8)
        ax0r.set_ylim(0,1); ax0r.set_yticks([])

        ax1r = axs[1].twinx()
        ax1r.plot(xs_a, after_probs, linewidth=0.7, linestyle=":", alpha=0.8)
        ax1r.set_ylim(0,1); ax1r.set_yticks([])

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True, help="window_eval.json BEFORE attack")
    p.add_argument("--after", required=True, help="window_eval.json AFTER attack")
    p.add_argument("--zero_fp", required=True, help="json with threshold (key thr or threshold)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--only_malware", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overlay_probs", action="store_true", help="also draw faint prob lines for debugging")
    args = p.parse_args()

    before = load_json(args.before)
    after = load_json(args.after)
    z = load_json(args.zero_fp)
    thr = get_threshold(z)

    outdir = Path(args.out_dir)
    per_dir = outdir / "per_sample"
    per_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # union of sample ids present in both files
    sids = sorted(set(before.get("samples",{}).keys()) & set(after.get("samples",{}).keys()), key=lambda x:int(x))
    if len(sids) == 0:
        raise SystemExit("No common sample ids between before/after files")

    for sid in sids:
        binfo = before["samples"].get(sid)
        ainfo = after["samples"].get(sid)
        if binfo is None or ainfo is None:
            continue
        label = int(binfo.get("label", 0))
        if args.only_malware and label != 1:
            continue

        b_scores = [float(x) for x in binfo.get("scores",[])]
        a_scores = [float(x) for x in ainfo.get("scores",[])]
        b_cls = scores_to_classes(b_scores, thr)
        a_cls = scores_to_classes(a_scores, thr)

        mal_before = sum(1 for v in b_cls if v==1)
        mal_after = sum(1 for v in a_cls if v==1)
        flipped = sum(1 for i in range(min(len(b_cls), len(a_cls))) if b_cls[i]==1 and a_cls[i]==0)

        png = per_dir / f"sample_{sid}_before_after.png"
        if args.overwrite or (not png.exists()):
            plot_classes(b_cls, a_cls, sid, label, str(png), thr, show_prob_overlay=args.overlay_probs, before_probs=b_scores, after_probs=a_scores)

        summary.append({
            "sample": sid,
            "label": label,
            "n_windows_before": len(b_scores),
            "n_windows_after": len(a_scores),
            "mal_windows_before": mal_before,
            "mal_windows_after": mal_after,
            "flipped_1_to_0": flipped
        })

    # write CSV & JSON summary
    out_csv = outdir / "comparison_summary.csv"
    if summary:
        keys = list(summary[0].keys())
    else:
        keys = ["sample","label","n_windows_before","n_windows_after","mal_windows_before","mal_windows_after","flipped_1_to_0"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in summary:
            writer.writerow(r)

    with open(outdir / "comparison_summary.json", "w", encoding="utf-8") as jf:
        json.dump({"threshold": thr, "n_samples": len(summary), "summary": summary}, jf, indent=2)

    print(f"[DONE] Wrote per-sample plots to {per_dir}")
    print(f"[DONE] Wrote CSV -> {out_csv}")
    print(f"[DONE] Wrote JSON -> {outdir / 'comparison_summary.json'}")
    print(f"[INFO] threshold used = {thr:.6f}")

if __name__ == "__main__":
    main()
