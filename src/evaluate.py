#!/usr/bin/env python3
"""
Apply a threshold on classifier outputs or a simple top-K mean rule and compute confusion matrix.
Usage:
  python src/evaluate.py --features results/sample_features_before.csv --model results/sample_clf_model.joblib --thr 0.959735
"""
import argparse, joblib, pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--model", required=False, help="path to model.joblib (scaler+clf) optional")
ap.add_argument("--thr", type=float, default=None, help="if using model, decision threshold on model proba")
ap.add_argument("--topk_mean_rule", type=int, default=0, help="if >0 use top-k mean rule and thr argument is compared to that mean")
ap.add_argument("--out", default="results/sample_rule_eval.json")
args = ap.parse_args()

df = pd.read_csv(args.features)
X = df.drop(columns=["sid","label"]).values
y = df["label"].values
res = {}
if args.topk_mean_rule > 0:
    k = args.topk_mean_rule
    preds = []
    for idx,row in df.iterrows():
        scs = sorted([row["max"], row["top3_mean"], row["top5_mean"]], reverse=True) # fallback
        # compute top-k mean from columns
        if k==1:
            val = row["top1"]
        elif k==3:
            val = row["top3_mean"]
        elif k==5:
            val = row["top5_mean"]
        else:
            val = row["max"]
        pred = 1 if val >= args.thr else 0
        preds.append(pred)
    tp=precision=recall=f1=0
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0,1]).ravel()
    res["confusion"] = {"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
else:
    if not args.model:
        raise SystemExit("Provide --model for classifier evaluation or use --topk_mean_rule")
    m = joblib.load(args.model)
    scaler = m["scaler"]
    clf = m["clf"]
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:,1]
    if args.thr is None:
        # default thr=0.5
        args.thr = 0.5
    preds = (probs >= args.thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0,1]).ravel()
    res["confusion"] = {"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
    res["metrics"] = {"precision": float(precision_score(y,preds)),"recall":float(recall_score(y,preds)),"f1":float(f1_score(y,preds))}
import json
with open(args.out,"w",encoding="utf-8") as f:
    json.dump(res,f,indent=2)
print("Wrote", args.out)
print("Confusion:", res["confusion"])
if "metrics" in res: print("Metrics:", res["metrics"])
