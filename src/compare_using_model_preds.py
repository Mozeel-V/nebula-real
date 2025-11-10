#!/usr/bin/env python3
"""
Compare BEFORE/AFTER using sample-level model probs.

Inputs:
  --before_we results/window_eval_before_vocabn/window_eval.json
  --after_we  results/window_eval_after_vocabn/window_eval.json
  --probs_before results/sample_probs_before.json
  --probs_after  results/sample_probs_after.json
  --thr 0.979159      # classifier probability threshold from tradeoff table
  --out_dir results/comparisons_model_based

Outputs:
  - comparison_summary.csv (sid,label,prob_before,prob_after,pred_before,pred_after,changed)
  - comparison_summary.json (aggregated metrics)
  - per_sample/*.png (before vs after window timelines)
"""
import argparse, json, csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_window_eval(path):
    return json.load(open(path,"r",encoding="utf-8")).get("samples",{})

def plot_timelines(before_scores, after_scores, sid, lab, out_dir):
    fig, ax = plt.subplots(figsize=(6,2.4))
    if before_scores:
        ax.plot(before_scores, label="Before", linewidth=0.7)
    if after_scores:
        ax.plot(after_scores, label="After", linewidth=0.7)
    ax.set_ylim(0,1)
    ax.set_title(f"SID={sid} label={lab}")
    ax.set_xlabel("Window idx")
    ax.set_ylabel("P(malware)")
    ax.legend(loc="upper right")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir)/f"{sid}.png", dpi=120)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_we", required=True)
    ap.add_argument("--after_we", required=True)
    ap.add_argument("--probs_before", required=True)
    ap.add_argument("--probs_after", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only_malware", action="store_true")
    args = ap.parse_args()

    before = load_window_eval(args.before_we)
    after = load_window_eval(args.after_we)
    probs_b = json.load(open(args.probs_before,"r",encoding="utf-8"))
    probs_a = json.load(open(args.probs_after,"r",encoding="utf-8"))

    sids = sorted(set(before.keys()) & set(after.keys()) & set(probs_b.keys()) & set(probs_a.keys()), key=lambda x:int(x))
    print("Comparing", len(sids), "samples")

    outdir = Path(args.out_dir)
    per_sample_dir = outdir/"per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    csvp = outdir/"comparison_summary.csv"
    jsonp = outdir/"comparison_summary.json"

    rows=[]
    y_true=[]; y_b=[]; y_a=[]

    for sid in sids:
        lab = int(before[sid].get("label",0))
        if args.only_malware and lab!=1:
            continue
        pb = float(probs_b[sid])
        pa = float(probs_a[sid])
        pred_b = 1 if pb >= args.thr else 0
        pred_a = 1 if pa >= args.thr else 0
        rows.append([sid, lab, pb, pa, pred_b, pred_a, int(pred_b != pred_a)])
        y_true.append(lab); y_b.append(pred_b); y_a.append(pred_a)
        # also save per-sample timeline plots
        before_scores = [float(x) for x in before[sid].get("scores",[])]
        after_scores = [float(x) for x in after[sid].get("scores",[])]
        try:
            plot_timelines(before_scores, after_scores, sid, lab, per_sample_dir)
        except Exception:
            pass

    # write CSV
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sid","label","prob_before","prob_after","pred_before","pred_after","changed"])
        writer.writerows(rows)

    # aggregated metrics
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, y_b, labels=[0,1]).ravel()
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, y_a, labels=[0,1]).ravel()
    metrics = {
        "before": {"tp":int(tp_b),"fp":int(fp_b),"tn":int(tn_b),"fn":int(fn_b)},
        "after":  {"tp":int(tp_a),"fp":int(fp_a),"tn":int(tn_a),"fn":int(fn_a)},
        "thr": args.thr
    }
    with open(jsonp, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=2)

    print("Wrote CSV:", csvp)
    print("Wrote JSON:", jsonp)
    print("Before metrics:", metrics["before"])
    print("After metrics:", metrics["after"])

if __name__ == "__main__":
    main()
