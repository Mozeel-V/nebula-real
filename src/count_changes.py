#!/usr/bin/env python3
"""
Count and summarize class changes between BEFORE and AFTER window_eval.json files.

Outputs:
 - out_dir/changes_per_sample.csv  (one row per sample with counts)
 - out_dir/changes_summary.json   (aggregate numbers)

Usage:
  python src/count_changes.py \
    --before results/window_eval_before/window_eval.json \
    --after  results/window_eval_after/window_eval.json \
    --zero_fp results/best_combo.json \
    --out_dir results/change_counts 
"""
import argparse, json, csv
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_threshold(z):
    for k in ("thr","threshold","thr_val","value"):
        if k in z:
            return float(z[k])
    if "thr" in z:
        return float(z["thr"])
    raise KeyError("no threshold key found in zero_fp JSON (expected 'thr' or 'threshold')")

def scores_to_classes(scores, thr):
    return [1 if float(s) >= thr - 1e-12 else 0 for s in scores]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--zero_fp", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--only_malware", action="store_true")
    args = p.parse_args()

    before = load_json(args.before)
    after = load_json(args.after)
    z = load_json(args.zero_fp)
    thr = get_threshold(z)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "changes_per_sample.csv"
    json_path = outdir / "changes_summary.json"

    sids = sorted(set(before.get("samples",{}).keys()) & set(after.get("samples",{}).keys()), key=lambda x:int(x))
    total_samples = 0
    samples_with_any_change = 0
    total_windows_before = 0
    total_windows_after = 0
    total_windows_changed = 0
    total_flipped_1_to_0 = 0
    total_flipped_0_to_1 = 0
    sample_rows = []

    for sid in sids:
        binfo = before["samples"].get(sid)
        ainfo = after["samples"].get(sid)
        if binfo is None or ainfo is None:
            continue
        label = int(binfo.get("label", 0))
        if args.only_malware and label != 1:
            continue
        total_samples += 1

        b_scores = [float(x) for x in binfo.get("scores",[])]
        a_scores = [float(x) for x in ainfo.get("scores",[])]

        b_cls = scores_to_classes(b_scores, thr)
        a_cls = scores_to_classes(a_scores, thr)

        nb = len(b_cls); na = len(a_cls)
        total_windows_before += nb
        total_windows_after += na

        minlen = min(nb, na)
        flipped_1_to_0 = sum(1 for i in range(minlen) if b_cls[i]==1 and a_cls[i]==0)
        flipped_0_to_1 = sum(1 for i in range(minlen) if b_cls[i]==0 and a_cls[i]==1)
        changed_windows = sum(1 for i in range(minlen) if b_cls[i] != a_cls[i])
        # account extra windows if lengths differ: those are considered "changed"
        if nb != na:
            extra = abs(nb - na)
            changed_windows += extra

        if changed_windows > 0:
            samples_with_any_change += 1

        total_flipped_1_to_0 += flipped_1_to_0
        total_flipped_0_to_1 += flipped_0_to_1
        total_windows_changed += changed_windows

        sample_rows.append({
            "sample": sid,
            "label": label,
            "n_windows_before": nb,
            "n_windows_after": na,
            "mal_windows_before": sum(b_cls),
            "mal_windows_after": sum(a_cls),
            "changed_windows": int(changed_windows),
            "flipped_1_to_0": int(flipped_1_to_0),
            "flipped_0_to_1": int(flipped_0_to_1),
            "any_changed": int(changed_windows>0)
        })

    # write CSV
    keys = ["sample","label","n_windows_before","n_windows_after","mal_windows_before","mal_windows_after","changed_windows","flipped_1_to_0","flipped_0_to_1","any_changed"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in sample_rows:
            writer.writerow(r)

    summary = {
        "threshold_used": thr,
        "n_samples_examined": total_samples,
        "samples_with_any_change": samples_with_any_change,
        "pct_samples_changed": (samples_with_any_change/total_samples*100) if total_samples>0 else 0.0,
        "total_windows_before": total_windows_before,
        "total_windows_after": total_windows_after,
        "total_windows_changed": total_windows_changed,
        "pct_windows_changed": (total_windows_changed / ((total_windows_before+total_windows_after)/2) * 100) if total_windows_before+total_windows_after>0 else 0.0,
        "total_flipped_1_to_0": total_flipped_1_to_0,
        "total_flipped_0_to_1": total_flipped_0_to_1
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    # print short summary
    print(f"Examined {total_samples} samples (only_malware={args.only_malware})")
    print(f"Samples with any window-level change: {samples_with_any_change} ({summary['pct_samples_changed']:.2f}%)")
    print(f"Total windows changed: {total_windows_changed} ({summary['pct_windows_changed']:.2f}% of windows)")
    print(f"Flipped 1->0 (evaded) : {total_flipped_1_to_0}")
    print(f"Flipped 0->1 (new alarm): {total_flipped_0_to_1}")
    print(f"Wrote per-sample CSV -> {csv_path}")
    print(f"Wrote summary JSON -> {json_path}")

if __name__ == "__main__":
    main()
