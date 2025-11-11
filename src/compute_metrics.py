#!/usr/bin/env python3
"""
Compute per-window metrics and per-sample (OR rule) metrics, before & after.
Writes results to out_dir/metrics_summary.json and per-window confusion CSV.
"""
import argparse, json, numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import csv

def load_we(path): return json.load(open(path,"r",encoding="utf-8"))["samples"]
def get_thr(zpath):
    z=json.load(open(zpath,"r",encoding="utf-8"))
    for k in ("thr","threshold","value"): 
        if k in z: return float(z[k])
    raise SystemExit("no thr in zero_fp json")

def flatten_window_labels(samples, thr):
    y_true=[]; y_pred=[]
    for sid,info in samples.items():
        lab=int(info.get("label",0))
        scores=[float(x) for x in info.get("scores",[])]
        for s in scores:
            y_true.append(lab)
            y_pred.append(1 if s>=thr else 0)
    return np.array(y_true), np.array(y_pred)

def sample_or_labels(samples, thr):
    sids=[]; y_true=[]; y_pred=[]
    for sid,info in samples.items():
        sids.append(sid)
        lab=int(info.get("label",0))
        scores=[float(x) for x in info.get("scores",[])]
        y_true.append(lab)
        y_pred.append(1 if any(s>=thr for s in scores) else 0)
    return sids, np.array(y_true), np.array(y_pred)

def write_confusion_csv(path, cm):
    tn, fp, fn, tp = cm.ravel()
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["tn","fp","fn","tp"])
        w.writerow([int(tn),int(fp),int(fn),int(tp)])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before_we", required=True); p.add_argument("--after_we", required=True)
    p.add_argument("--zero_fp", required=True); p.add_argument("--out_dir", required=True)
    args=p.parse_args()
    thr=get_thr(args.zero_fp)
    before=load_we(args.before_we); after=load_we(args.after_we)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)

    # per-window
    yb_true,yb_pred = flatten_window_labels(before, thr)
    ya_true,ya_pred = flatten_window_labels(after, thr)

    cm_b = confusion_matrix(yb_true,yb_pred,labels=[0,1])
    cm_a = confusion_matrix(ya_true,ya_pred,labels=[0,1])

    write_confusion_csv(Path(args.out_dir)/"window_confusion_before.csv", cm_b)
    write_confusion_csv(Path(args.out_dir)/"window_confusion_after.csv", cm_a)

    # compute metrics
    def metrics(y_true,y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
        return {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
                "accuracy":float(accuracy_score(y_true,y_pred)),
                "precision":float(precision_score(y_true,y_pred,zero_division=0)),
                "recall":float(recall_score(y_true,y_pred,zero_division=0)),
                "f1":float(f1_score(y_true,y_pred,zero_division=0))}

    wm_before = metrics(yb_true,yb_pred)
    wm_after  = metrics(ya_true,ya_pred)

    # per-sample OR rule
    sids_b, yb_s_true, yb_s_pred = sample_or_labels(before, thr)
    sids_a, ya_s_true, ya_s_pred = sample_or_labels(after, thr)

    sm_before = metrics(yb_s_true, yb_s_pred)
    sm_after  = metrics(ya_s_true, ya_s_pred)

    # evasion: malware that flip from predicted=1 before -> predicted=0 after
    # map sids to preds
    pred_b_map = {sid:int(any(float(x)>=thr for x in before[sid].get("scores",[]))) for sid in before}
    pred_a_map = {sid:int(any(float(x)>=thr for x in after[sid].get("scores",[])))  for sid in after}
    evaded = []
    for sid,info in before.items():
        lab=int(info.get("label",0))
        if lab==1 and pred_b_map.get(sid,0)==1 and pred_a_map.get(sid,0)==0:
            evaded.append(sid)

    summary = {"threshold":thr,
               "window_metrics": {"before":wm_before,"after":wm_after},
               "sample_metrics_or_rule": {"before":sm_before,"after":sm_after},
               "n_samples_examined": len(set(before.keys()) & set(after.keys())),
               "n_evaded_samples": len(evaded),
               "evaded_sample_ids": evaded[:200]}

    with open(Path(args.out_dir)/"metrics_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print("Wrote metrics_summary.json ->", Path(args.out_dir)/"metrics_summary.json")
    print("Window before:", wm_before)
    print("Sample OR before:", sm_before)
    print("Window after:", wm_after)
    print("Sample OR after:", sm_after)
    print("Evaded samples count:", len(evaded))

if __name__=="__main__":
    main()
