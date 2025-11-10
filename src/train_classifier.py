#!/usr/bin/env python3
"""
Train logistic regression on sample features and search for validation thresholds that give FP==0.
Outputs:
 - results/sample_clf_results.json (summary)
 - results/sample_clf_model.joblib (trained final model on full data)
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="CSV from extract_sample_features.py")
ap.add_argument("--out_prefix", default="results/sample_clf")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

df = pd.read_csv(args.features)
X = df.drop(columns=["sid","label"]).values
y = df["label"].values
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

results = []
best_overall = {"tp":-1, "thr":None, "fold":None}
fold_id = 0
# store per-fold info
per_fold = []

for train_idx, val_idx in skf.split(X,y):
    fold_id += 1
    Xtr, Xv = X[train_idx], X[val_idx]
    ytr, yv = y[train_idx], y[val_idx]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xv_s = scaler.transform(Xv)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    clf.fit(Xtr_s, ytr)
    probs = clf.predict_proba(Xv_s)[:,1]
    # search thresholds on unique probs sorted descending
    uniq = np.unique(np.round(probs,6))
    uniq_sorted = np.sort(uniq)
    best = {"thr":None,"tp":-1,"fp":9999,"tn":0,"fn":0}
    for thr in uniq_sorted:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, preds, labels=[0,1]).ravel()
        if fp==0 and tp > best["tp"]:
            best = {"thr":float(thr),"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
    # if no thr yields fp==0, we record best with minimal fp then highest tp
    if best["thr"] is None:
        # find minimal fp
        cand = []
        for thr in uniq_sorted:
            preds = (probs >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yv, preds, labels=[0,1]).ravel()
            cand.append((fp, -tp, thr, tp, fp, tn, fn))
        cand_sorted = sorted(cand, key=lambda x:(x[0], x[1]))
        fp, nnegtp, thr, tp, fp, tn, fn = cand_sorted[0]
        best = {"thr":float(thr),"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
    per_fold.append({"fold":fold_id,"best":best})
    # track best across folds
    if best["tp"] > best_overall["tp"]:
        best_overall = {"tp":best["tp"], "thr":best["thr"], "fold":fold_id}

# Train final model on full data and save
scaler_full = StandardScaler().fit(X)
X_s_full = scaler_full.transform(X)
clf_full = LogisticRegression(max_iter=2000, solver="liblinear")
clf_full.fit(X_s_full, y)
Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"scaler":scaler_full, "clf":clf_full}, args.out_prefix + "_model.joblib")
summary = {"per_fold": per_fold, "best_overall": best_overall}
with open(args.out_prefix + "_results.json","w",encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Wrote:", args.out_prefix + "_results.json and model.joblib")
