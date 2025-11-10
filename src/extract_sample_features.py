#!/usr/bin/env python3
"""
Extract per-sample features from window_eval.json
Outputs CSV with: sid,label,n_windows,max,mean,median,std,top1,top3_mean,top5_mean,frac_above_0.3,frac_above_0.35,...
"""
import json, csv, argparse, numpy as np
from pathlib import Path

def topk_mean(arr,k):
    if not arr: return 0.0
    a = sorted(arr, reverse=True)[:k]
    return float(np.mean(a))

def frac_above(arr,thr):
    if not arr: return 0.0
    return float(sum(1 for x in arr if x>=thr))/len(arr)

ap = argparse.ArgumentParser()
ap.add_argument("--we", required=True, help="window_eval.json")
ap.add_argument("--out", default="results/sample_features.csv")
args = ap.parse_args()

data = json.load(open(args.we,"r",encoding="utf-8"))
samples = data.get("samples",{})

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out,"w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["sid","label","n_windows","max","mean","median","std","top1","top3_mean","top5_mean",
              "frac_above_0.30","frac_above_0.35","frac_above_0.40"]
    writer.writerow(header)
    for sid,info in samples.items():
        sc = [float(x) for x in info.get("scores",[])]
        if len(sc)==0:
            arr = [0.0]*len(header)
            writer.writerow([sid,int(info.get("label",0)),0]+[0.0]*(len(header)-3))
            continue
        n = len(sc)
        mx = max(sc)
        mean = float(np.mean(sc))
        med = float(np.median(sc))
        std = float(np.std(sc))
        t1 = float(sorted(sc,reverse=True)[0])
        t3 = topk_mean(sc,3)
        t5 = topk_mean(sc,5)
        f30 = frac_above(sc,0.30)
        f35 = frac_above(sc,0.35)
        f40 = frac_above(sc,0.40)
        writer.writerow([sid,int(info.get("label",0)),n,mx,mean,med,std,t1,t3,t5,f30,f35,f40])

print("Wrote features ->", args.out)
