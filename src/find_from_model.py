#!/usr/bin/env python3
import joblib, numpy as np, pandas as pd, json, argparse
from sklearn.metrics import confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", default="results/sample_threshold_candidates.json")
args = ap.parse_args()

df = pd.read_csv(args.features)
X = df.drop(columns=["sid","label"]).values
y = df["label"].values
m = joblib.load(args.model)
scaler = m["scaler"]; clf = m["clf"]
probs = clf.predict_proba(scaler.transform(X))[:,1]

uniq = np.unique(np.round(probs,6))
uniq_sorted = np.sort(uniq)
best = None
cands = []
for thr in uniq_sorted:
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0,1]).ravel()
    cands.append({"thr":float(thr),"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)})
    if fp == 0:
        if best is None or int(tp) > int(best["tp"]):
            best = {"thr":float(thr),"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
# additionally keep a small list of best with small FP
cands_sorted = sorted(cands, key=lambda x: (x["fp"], -x["tp"]))
out = {"best_zero_fp": best, "top_candidates": cands_sorted[:50]}
open(args.out,"w").write(json.dumps(out, indent=2))
print("Wrote", args.out)
print("Best zero-FP:", best)
print("Top few (by fp asc, tp desc):")
for x in cands_sorted[:10]:
    print(x)
