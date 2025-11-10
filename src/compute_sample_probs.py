#!/usr/bin/env python3
"""
Compute sample-level probabilities from saved scaler+clf (joblib) for a features CSV.

Input:
  --features results/sample_features_before.csv
  --model results/sample_clf_model.joblib

Output:
  results/sample_probs_before.json  (mapping sid -> prob)
"""
import argparse, joblib, json, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()

df = pd.read_csv(args.features)
sids = df["sid"].astype(str).tolist()
X = df.drop(columns=["sid","label"]).values

m = joblib.load(args.model)
scaler = m["scaler"]; clf = m["clf"]
probs = clf.predict_proba(scaler.transform(X))[:,1]

outp = args.out or (Path(args.features).parent / (Path(args.features).stem + ".probs.json"))
outp = str(outp)
d = {sid: float(p) for sid,p in zip(sids, probs)}
Path(outp).parent.mkdir(parents=True, exist_ok=True)
with open(outp, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)
print("Wrote", outp)
