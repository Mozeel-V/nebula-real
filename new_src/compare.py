#!/usr/bin/env python3
"""
compare_before_after_metrics.py

Generate metrics and side-by-side plots (before vs after) grouped into
malware / benign folders. Saves metrics_summary.json, per_sample_summary.csv,
evaded_samples.json, and per-sample PNGs.

Usage example:
 python src/eval/compare_before_after_metrics.py \
   --before results/window_eval_before/window_eval.json \
   --after  results/window_eval_after/window_eval.json \
   --out_dir results/comparisons_full \
   --thr 0.911801288443 \
   --minwins 0 --consecutive_k 0 --save_side_by_side
"""
import argparse, json, csv, math, os, sys
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pprint import pprint
from typing import List

def load_we(path):
    j = json.load(open(path, "r", encoding="utf-8"))
    if "samples" in j:
        return j["samples"], j.get("meta", {})
    return j, {}

def sample_pred_from_scores(scores: List[float], thr: float, minwins:int=0, consecutive_k:int=0):
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

def window_level_metrics_all(samples, thr):
    # compute window-level tp,fp,tn,fn counting each window across all samples
    tp = fp = tn = fn = 0
    for sid, info in samples.items():
        y = int(info["label"])
        for s in info["scores"]:
            pred = 1 if s >= thr else 0
            if y == 1 and pred == 1:
                tp += 1
            elif y == 1 and pred == 0:
                fn += 1
            elif y == 0 and pred == 1:
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def sample_level_metrics(samples, thr, minwins=0, consecutive_k=0):
    tp = fp = tn = fn = 0
    per_sample = {}
    for sid, info in samples.items():
        y = int(info["label"])
        pred = sample_pred_from_scores(info["scores"], thr, minwins=minwins, consecutive_k=consecutive_k)
        per_sample[sid] = {"label": y, "pred": pred, "n_windows": len(info["scores"]), "n_mal_windows": sum(1 for s in info["scores"] if s>=thr)}
        if y == 1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}, per_sample

def precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = (tp + 0.0 + 0)  # placeholder
    return {"precision": prec, "recall": rec, "f1": f1}

def plot_side_by_side(before_scores, after_scores, thr, out_png, title=None):
    """
    Plot side-by-side *class* timelines (not probabilities).
    Converts scores -> class: 1 if score >= thr else 0.
    Saves horizontal left|right figure to out_png.
    """
    # convert to binary timelines
    b_cls = [1 if s >= thr else 0 for s in before_scores]
    a_cls = [1 if s >= thr else 0 for s in after_scores]

    # ensure same x-range for visual comparsion
    L = max(len(b_cls), len(a_cls))
    # pad shorter with zeros (benign) to align x axis
    if len(b_cls) < L:
        b_cls = b_cls + [0] * (L - len(b_cls))
    if len(a_cls) < L:
        a_cls = a_cls + [0] * (L - len(a_cls))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    ax1, ax2 = axes

    ax1.step(range(len(b_cls)), b_cls, where='mid', linewidth=2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["benign", "malware"])
    ax1.set_xlim(0, L-1)
    ax1.set_xlabel("window index")
    ax1.set_title("Before" + ((" — " + title) if title else ""))

    ax2.step(range(len(a_cls)), a_cls, where='mid', linewidth=2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["", ""])   # hide redundant ytick labels on right panel
    ax2.set_xlim(0, L-1)
    ax2.set_xlabel("window index")
    ax2.set_title("After" + ((" — " + title) if title else ""))

    # draw thin horizontal separating lines for visibility
    for ax in (ax1, ax2):
        ax.grid(False)
        ax.set_facecolor("white")

    plt.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def ensure_dirs(base:Path):
    for cls in ("malware","benign"):
        for name in ("before","after","side_by_side"):
            (base/cls/name).mkdir(parents=True, exist_ok=True)

def write_json(path, obj):
    open(path, "w", encoding="utf-8").write(json.dumps(obj, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thr", type=float, default=None, help="explicit threshold")
    ap.add_argument("--zero_fp", type=str, default=None, help="json file containing {'thr':...}")
    ap.add_argument("--minwins", type=int, default=0)
    ap.add_argument("--consecutive_k", type=int, default=0)
    ap.add_argument("--save_side_by_side", action="store_true", help="save side-by-side plots")
    ap.add_argument("--max_examples_plot", type=int, default=200, help="max side-by-side plots per class (to limit runtime)")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(outdir)

    # load we files
    before_samples, before_meta = load_we(args.before)
    after_samples, after_meta = load_we(args.after)

    # sanity: same samples
    s_before = set(before_samples.keys())
    s_after = set(after_samples.keys())
    if s_before != s_after:
        print("[warn] sample ID sets differ between before and after. Using intersection.", file=sys.stderr)
    sids = sorted(list(s_before & s_after))
    print(f"[info] samples to compare: {len(sids)}")

    # pick thr
    thr = args.thr
    if thr is None and args.zero_fp:
        try:
            z = json.load(open(args.zero_fp, "r", encoding="utf-8"))
            thr = float(z.get("thr") or z.get("threshold") or z.get("value"))
            print("[info] loaded thr from zero_fp:", thr)
        except Exception as e:
            print("[warn] could not read zero_fp json:", e)
    if thr is None:
        # fallback: median of per-sample-max from before
        per_max = [max(before_samples[sid]["scores"]) if len(before_samples[sid]["scores"])>0 else 0.0 for sid in sids]
        thr = float(sorted(per_max)[max(0, int(0.5*len(per_max)) )])  # median
        print("[info] no thr provided, using median of maxima:", thr)
    else:
        print("[info] using thr =", thr)

    # compute window-level metrics
    window_metrics_before = window_level_metrics_all({sid: before_samples[sid] for sid in sids}, thr)
    window_metrics_after  = window_level_metrics_all({sid: after_samples[sid] for sid in sids}, thr)

    # compute sample-level metrics (OR rule/minwins/consec)
    sample_metrics_before, per_sample_before = sample_level_metrics({sid: before_samples[sid] for sid in sids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)
    sample_metrics_after, per_sample_after  = sample_level_metrics({sid: after_samples[sid]  for sid in sids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)

    # compute metrics per-class (malware vs benign)
    malware_ids = [sid for sid in sids if int(before_samples[sid]["label"]) == 1]
    benign_ids  = [sid for sid in sids if int(before_samples[sid]["label"]) == 0]
    m_before_metrics, m_per = sample_level_metrics({sid: before_samples[sid] for sid in malware_ids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)
    m_after_metrics, m_per_after = sample_level_metrics({sid: after_samples[sid] for sid in malware_ids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)
    b_before_metrics, b_per = sample_level_metrics({sid: before_samples[sid] for sid in benign_ids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)
    b_after_metrics, b_per_after = sample_level_metrics({sid: after_samples[sid] for sid in benign_ids}, thr, minwins=args.minwins, consecutive_k=args.consecutive_k)

    # compute evaded samples (malware that was pred=1 before and pred=0 after)
    evaded = []
    for sid in malware_ids:
        prev = per_sample_before.get(sid, {})
        post = per_sample_after.get(sid, {})
        if prev.get("pred", None) == 1 and post.get("pred", None) == 0:
            evaded.append(sid)
    evaded_count = len(evaded)

    # write per-sample CSV
    csvp = outdir / "per_sample_summary.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["sid","label","n_windows","n_mal_windows_before","pred_before","n_mal_windows_after","pred_after","changed"])
        for sid in sids:
            lb = int(before_samples[sid]["label"])
            nb = len(before_samples[sid]["scores"])
            nmb = sum(1 for s in before_samples[sid]["scores"] if s >= thr)
            pred_b = per_sample_before[sid]["pred"]
            nma = len(after_samples[sid]["scores"])
            nma_mal = sum(1 for s in after_samples[sid]["scores"] if s >= thr)
            pred_a = per_sample_after[sid]["pred"]
            changed = 1 if pred_b != pred_a else 0
            writer.writerow([sid, lb, nb, nmb, pred_b, nma_mal, pred_a, changed])
    print("[info] wrote per-sample CSV:", csvp)

    # write window metrics and sample metrics summary
    summary = {
        "thr_used": thr,
        "n_samples": len(sids),
        "window_before": window_metrics_before,
        "window_after": window_metrics_after,
        "sample_or_before": sample_metrics_before,
        "sample_or_after": sample_metrics_after,
        "malware_sample_before": m_before_metrics,
        "malware_sample_after": m_after_metrics,
        "benign_sample_before": b_before_metrics,
        "benign_sample_after": b_after_metrics,
        "evaded_count": evaded_count,
        "evaded_samples": evaded,
    }
    write_json(outdir/"metrics_summary.json", summary)
    print("[info] wrote metrics_summary:", outdir/"metrics_summary.json")

    # save evaded samples separately
    write_json(outdir/"evaded_samples.json", evaded)
    print("[info] evaded samples count:", evaded_count)

    # produce per-class side-by-side plots if requested
    if args.save_side_by_side:
        max_plot = int(args.max_examples_plot)
        # malware
        wrote_m = 0
        for sid in malware_ids:
            if wrote_m >= max_plot: break
            b_scores = before_samples[sid]["scores"]
            a_scores = after_samples[sid]["scores"]
            title = f"SID {sid} label=1"
            out_png = outdir / "malware" / "side_by_side" / f"{sid}.png"
            plot_side_by_side(b_scores, a_scores, thr, out_png, title=title)
            wrote_m += 1
        print(f"[info] saved {wrote_m} malware side-by-side plots to {outdir/'malware'/'side_by_side'}")

        # benign
        wrote_b = 0
        for sid in benign_ids:
            if wrote_b >= max_plot: break
            b_scores = before_samples[sid]["scores"]
            a_scores = after_samples[sid]["scores"]
            title = f"SID {sid} label=0"
            out_png = outdir / "benign" / "side_by_side" / f"{sid}.png"
            plot_side_by_side(b_scores, a_scores, thr, out_png, title=title)
            wrote_b += 1
        print(f"[info] saved {wrote_b} benign side-by-side plots to {outdir/'benign'/'side_by_side'}")

    # also save separated single-pane plots (before/after) into class folders for quick browsing
    for sid in sids:
        lb = int(before_samples[sid]["label"])
        cls = "malware" if lb==1 else "benign"
        # before
        p_before = outdir / cls / "before" / f"{sid}.png"
        plt.figure(figsize=(6,1.8))
        plt.step(range(len(before_samples[sid]["scores"])), before_samples[sid]["scores"], where="mid")
        plt.ylim(-0.02,1.02); plt.axhline(thr, color='red', linewidth=0.6)
        plt.title(f"SID {sid} before pred={per_sample_before[sid]['pred']} n_malw={per_sample_before[sid]['n_mal_windows']}")
        plt.tight_layout(); plt.savefig(p_before, dpi=100); plt.close()
        # after
        p_after = outdir / cls / "after" / f"{sid}.png"
        plt.figure(figsize=(6,1.8))
        plt.step(range(len(after_samples[sid]["scores"])), after_samples[sid]["scores"], where="mid")
        plt.ylim(-0.02,1.02); plt.axhline(thr, color='red', linewidth=0.6)
        plt.title(f"SID {sid} after  pred={per_sample_after[sid]['pred']} n_malw={per_sample_after[sid]['n_mal_windows']}")
        plt.tight_layout(); plt.savefig(p_after, dpi=100); plt.close()

    print("[done] All outputs written to:", outdir)
    # print compact summary
    pprint(summary)

if __name__ == "__main__":
    main()
