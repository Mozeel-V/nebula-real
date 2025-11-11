#!/usr/bin/env python3
"""
Create sample-level labels using rule: sample_malware = (any window_malware == 1)
Also saves per-sample PNGs into directories:
 out_root/malware/before, out_root/malware/after, out_root/goodware/before, out_root/goodware/after
"""
import argparse, json, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_we(path): return json.load(open(path,"r",encoding="utf-8"))["samples"]
def scores_to_classes(scores, thr): return [1 if s >= thr else 0 for s in scores]

def save_sample_plot(cls, sid, outpath, title):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.step(range(len(cls)), cls, where="mid")
    ax.set_ylim(-0.1,1.1); ax.set_yticks([0,1]); ax.set_yticklabels(["goodware","malware"])
    ax.set_xlabel("window index"); ax.set_title(title)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before_we", required=True)
    p.add_argument("--after_we", required=True)
    p.add_argument("--zero_fp", required=True)
    p.add_argument("--out_root", default="results/classes_split")
    p.add_argument("--only_malware", action="store_true")
    args=p.parse_args()

    z = json.load(open(args.zero_fp,"r",encoding="utf-8"))
    thr = None
    for k in ("thr","threshold","value"): 
        if k in z: thr=float(z[k]); break
    if thr is None: raise SystemExit("no thr in zero_fp json")

    before = load_we(args.before_we)
    after  = load_we(args.after_we)
    sids = sorted(set(before.keys()) & set(after.keys()), key=lambda x:int(x))

    out_root = Path(args.out_root)
    for sid in sids:
        b = before[sid]; a = after[sid]
        label = int(b.get("label",0))
        if args.only_malware and label!=1: continue

        b_scores = [float(x) for x in b.get("scores",[])]
        a_scores = [float(x) for x in a.get("scores",[])]

        b_cls = scores_to_classes(b_scores, thr)
        a_cls = scores_to_classes(a_scores, thr)

        sample_label_before = 1 if any(b_cls) else 0
        sample_label_after  = 1 if any(a_cls) else 0

        # folder selection and saving
        folder_before = ("malware" if sample_label_before==1 else "goodware")
        folder_after  = ("malware" if sample_label_after==1 else "goodware")

        save_sample_plot(b_cls, sid, out_root/f"{folder_before}/before/sample_{sid}.png", f"SID {sid} before")
        save_sample_plot(a_cls, sid, out_root/f"{folder_after}/after/sample_{sid}.png", f"SID {sid} after")

    # Also write a CSV of sample-level results
    csv_path = out_root/"sample_level_labels.csv"
    rows=[]
    for sid in sids:
        b = before[sid]; a = after[sid]; label=int(b.get("label",0))
        b_cls=[1 if float(x)>=thr else 0 for x in b.get("scores",[])]
        a_cls=[1 if float(x)>=thr else 0 for x in a.get("scores",[])]
        rows.append({"sid":sid,"label":label,
                     "sample_before": int(any(b_cls)),
                     "sample_after":  int(any(a_cls)),
                     "mal_windows_before": sum(b_cls),
                     "mal_windows_after":  sum(a_cls)})
    outroot=Path(args.out_root); outroot.mkdir(parents=True,exist_ok=True)
    with open(outroot/"sample_level_labels.csv","w",encoding="utf-8",newline="") as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("Wrote sample split & CSV ->", outroot)

if __name__=="__main__":
    main()
