#!/usr/bin/env python3
import joblib, numpy as np, pandas as pd, json, argparse
from sklearn.metrics import confusion_matrix
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--max_fp", type=int, default=20)
ap.add_argument("--out", default="results/fp_tp_tradeoff.json")
args = ap.parse_args()

df = pd.read_csv(args.features)
X = df.drop(columns=["sid","label"]).values
y = df["label"].values
m = joblib.load(args.model)
scaler = m["scaler"]; clf = m["clf"]
probs = clf.predict_proba(scaler.transform(X))[:,1]

uniq = np.unique(np.round(probs,6))
uniq_sorted = np.sort(uniq)
trade = {}
for thr in uniq_sorted:
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0,1]).ravel()
    if fp <= args.max_fp:
        trade.setdefault(int(fp), []).append({"thr":float(thr),"tp":int(tp),"tn":int(tn),"fn":int(fn)})

# For each fp, keep best tp
best_per_fp = {}
for fp, lst in trade.items():
    best = max(lst, key=lambda x: x["tp"])
    best_per_fp[int(fp)] = best

# convert keys to strings to be safe for JSON
best_per_fp_str_keys = {str(k): v for k, v in best_per_fp.items()}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
open(args.out,"w",encoding="utf-8").write(json.dumps(best_per_fp_str_keys, indent=2))
print("Wrote", args.out)
print("Best per allowed FP (fp -> best entry):")
for k in sorted(best_per_fp.keys()):
    print(k, best_per_fp[k])
