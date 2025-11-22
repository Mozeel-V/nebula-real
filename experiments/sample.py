#!/usr/bin/env python3
"""
Creates multiple random balanced subsets (1000 goodware + 1000 malware),
train model on each subset, run window-eval, find zero-FP threshold,
and collect summary metrics.

Usage:
python new_src/experiments/sample.py \
  --full_tsv data/dataset_small_2k_normalized.tsv \
  --n_runs 3 \
  --out_dir experiments/multi_runs \
  --train_script new_src/train_supervised.py \
  --window_eval_script new_src/window_eval_plot.py \
  --find_zfp_script new_src/find_zero_fp.py \
  --vocab checkpoints/vocab_n.json \
  --epochs 6 \
  --batch_size 64
"""

import argparse, random, csv, json, os, subprocess, time
from pathlib import Path
from collections import defaultdict

def read_full_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln: continue
            parts = ln.split("\t", 2)
            if len(parts) < 3: continue
            sid, lab, trace = parts[0], parts[1], parts[2]
            rows.append((sid, int(lab), trace))
    return rows

def sample_balanced(rows, seed, n_per_class=1000):
    random.seed(seed)
    mal = [r for r in rows if r[1]==1]
    good = [r for r in rows if r[1]==0]
    if len(mal) < n_per_class or len(good) < n_per_class:
        raise RuntimeError(f"Not enough samples: mal={len(mal)} good={len(good)} need={n_per_class}")
    mal_s = random.sample(mal, n_per_class)
    good_s = random.sample(good, n_per_class)
    combined = mal_s + good_s
    random.shuffle(combined)
    return combined

def write_tsv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for sid, lab, trace in rows:
            fo.write(f"{sid}\t{lab}\t{trace}\n")

def run_cmd(cmd, cwd=None, env=None):
    print("[run]"," ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")

def try_read_json(path):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_tsv", required=True, help="Full normalized TSV with sid<TAB>label<TAB>trace")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--out_dir", default="experiments/multi_runs")
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    # script paths / CLI
    p.add_argument("--train_script", default="new_src/train_supervised.py")
    p.add_argument("--window_eval_script", default="new_src/window_eval_plot.py")
    p.add_argument("--find_zfp_script", default="new_src/find_zero_fp.py")
    p.add_argument("--vocab", default="checkpoints/vocab_n.json")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight0", type=float, default=1.0)
    p.add_argument("--weight1", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train_cmd_extra", default="", help="Extra args for training (quoted)")
    p.add_argument("--eval_cmd_extra", default="", help="Extra args for eval (quoted)")
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_full_tsv(args.full_tsv)
    print(f"[info] loaded full dataset rows={len(rows)}")

    summary_rows = []

    for run in range(args.n_runs):
        run_seed = args.seed + run
        print(f"\n=== RUN {run+1}/{args.n_runs} seed={run_seed} ===")
        subset = sample_balanced(rows, seed=run_seed, n_per_class=args.n_per_class)
        tsv_path = outdir / f"subset_{run+1}_{run_seed}.tsv"
        write_tsv(subset, tsv_path)
        print("[info] wrote subset", tsv_path)

        # train output dir
        run_ckpt_dir = outdir / f"ckpt_run_{run+1}"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        # TRAIN
        train_cmd = [
            "python", args.train_script,
            "--data", str(tsv_path),
            "--vocab", args.vocab,
            "--out_dir", str(run_ckpt_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight0", str(args.weight0),
            "--weight1", str(args.weight1),
            "--model_type", "nebula_cls",
        ]
        if args.train_cmd_extra:
            train_cmd += args.train_cmd_extra.strip().split()
        t0 = time.time()
        run_cmd(train_cmd)
        t_train = time.time() - t0
        # assume checkpoint best.pt written to run_ckpt_dir/best.pt
        ckpt_path = run_ckpt_dir / "best.pt"
        if not ckpt_path.exists():
            # try common alt
            ckpt_path = run_ckpt_dir / "checkpoint.pt"
        print("[info] ckpt path:", ckpt_path)

        # WINDOW EVAL (before/after for this run - here we run BEFORE only for baseline)
        we_out_before = outdir / f"window_eval_before_run_{run+1}.json"
        eval_cmd = [
            "python", args.window_eval_script,
            "--data_file", str(tsv_path),
            "--ckpt", str(ckpt_path),
            "--vocab", args.vocab,
            "--out_dir", str(outdir / f"window_eval_before_run_{run+1}"),
            "--window_unit", "event",
            "--events_per_window", "16",
            "--stride_events", "4",
        ]
        if args.eval_cmd_extra:
            eval_cmd += args.eval_cmd_extra.strip().split()
        t0 = time.time()
        run_cmd(eval_cmd)
        t_eval = time.time() - t0
        we_json = outdir / f"window_eval_before_run_{run+1}" / "window_eval.json"

        # compute zero-FP threshold for this run using find_zero_fp_from_we.py
        zfp_outdir = outdir / f"zfp_run_{run+1}"
        zfp_outdir.mkdir(parents=True, exist_ok=True)
        zfp_cmd = ["python", args.find_zfp_script, "--we", str(we_json), "--out_dir", str(zfp_outdir), "--grid_samples", "500", "--max_minwins", "3", "--max_consec", "2", "--debug"]
        run_cmd(zfp_cmd)
        zfp_json = zfp_outdir / "best_combo.json"
        zfp = try_read_json(zfp_json)
        thr_used = None
        if zfp and "best_combo" in zfp:
            thr_used = zfp["best_combo"].get("thr")
        print("[info] thr_used:", thr_used)

        # gather simple metrics from window_eval.json
        we_data = try_read_json(we_json)
        n_samples = 0
        malware_sample_metrics = {}
        if we_data and "samples" in we_data:
            n_samples = len(we_data["samples"])
            # compute simple per-sample OR recall (using thr if found)
            tp = fp = tn = fn = 0
            if thr_used is not None:
                for sid, info in we_data["samples"].items():
                    lab = int(info["label"])
                    scores = info.get("scores", [])
                    pred = 1 if any(s >= thr_used for s in scores) else 0
                    if lab==1 and pred==1: tp+=1
                    if lab==1 and pred==0: fn+=1
                    if lab==0 and pred==1: fp+=1
                    if lab==0 and pred==0: tn+=1
            malware_sample_metrics = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

        summary_rows.append({
            "run": run+1,
            "seed": run_seed,
            "subset_tsv": str(tsv_path),
            "ckpt_dir": str(run_ckpt_dir),
            "ckpt": str(ckpt_path),
            "we_json": str(we_json),
            "thr_used": thr_used,
            "train_time_sec": round(t_train,2),
            "eval_time_sec": round(t_eval,2),
            "n_samples_eval": n_samples,
            "malware_sample_tp": malware_sample_metrics.get("tp", None),
            "malware_sample_fp": malware_sample_metrics.get("fp", None),
            "malware_sample_tn": malware_sample_metrics.get("tn", None),
            "malware_sample_fn": malware_sample_metrics.get("fn", None)
        })

    # write summary CSV
    csvp = outdir / "multi_sample_summary.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as cf:
        import csv
        fields = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print("[done] wrote summary:", csvp)

if __name__ == "__main__":
    main()
