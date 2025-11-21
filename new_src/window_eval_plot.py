#!/usr/bin/env python3
"""
Enhanced window evaluation script.

Usage example:
python src/eval/window_eval_plot.py \
  --data_file data/dataset_small_2k_normalized.tsv \
  --ckpt checkpoints/run_weighted_w0_8/best.pt \
  --vocab checkpoints/vocab_n.json \
  --out_dir results/window_eval_weighted \
  --events_per_window 16 --stride_events 4 --batch_size 128 \
  --thr 0.979159 --minwins 2 --split_dirs

Outputs:
 - out_dir/window_eval.json
 - out_dir/window_eval.score_hist.png
 - out_dir/summaries/*.csv
 - optionally: out_dir/classes_split/{malware,goodware}/sample_<sid>.png
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict
import csv
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from tokenizer import Tokenizer
from nebula_model import NebulaTiny, NebulaCLS

# -------------------------
# Utilities
# -------------------------
def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    # ck could be dict with 'model' key or be the state_dict itself
    state = None
    config = None
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        config = ck.get("config", {})
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
        config = ck.get("config", {})
    else:
        # assume ck itself is a state_dict
        # try to infer config from file name or parent
        state = ck
        config = {}

    return state, config

def build_model_from_ck(config, vocab_size, device):
    # prefer explicit config keys if present
    mtype = config.get("model_type", None)
    d_model = int(config.get("d_model", 128))
    nhead = int(config.get("nhead", 4))
    num_layers = int(config.get("num_layers", 2))
    ff = int(config.get("ff", config.get("dim_feedforward", 256)))
    max_len = int(config.get("max_len", 256))

    # default to CLS variant if num_layers >=4 or config asks nebula_cls
    if mtype == "nebula_cls" or num_layers >= 4:
        model = NebulaCLS(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=max_len)
    else:
        model = NebulaTiny(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=max_len)
    model.to(device)
    return model

def robust_build_and_load_model(state_dict, ck_cfg, vocab_size, device):
    """
    Build NebulaCLS or NebulaTiny depending on checkpoint content,
    adapt positional encodings if needed, and load state dict robustly.
    Returns the model (on device).
    """
    # infer config entries (fallbacks)
    d_model = int(ck_cfg.get("d_model", ck_cfg.get("hidden_size", 128)))
    nhead = int(ck_cfg.get("nhead", 4))
    num_layers = int(ck_cfg.get("num_layers", ck_cfg.get("layers", 2)))
    ff = int(ck_cfg.get("ff", ck_cfg.get("dim_feedforward", 256)))
    # max_len inference: either in config or from pos.pe in state_dict
    ck_pe = None
    if "pos.pe" in state_dict:
        ck_pe = state_dict["pos.pe"]
    else:
        for k in state_dict.keys():
            if k.endswith("pe") or k.endswith("pos.pe"):
                ck_pe = state_dict[k]
                break

    ck_has_cls = any(k.endswith("cls_token") or k == "cls_token" for k in state_dict.keys()) or ck_cfg.get("model_type","") == "nebula_cls"
    # infer checkpoint max_len from ck_pe if present
    ck_max_len = None
    if ck_pe is not None:
        try:
            ck_max_len = ck_pe.shape[1]  # shape [1, L, D]
        except Exception:
            ck_max_len = None

    # if checkpoint had cls token, then ck_max_len likely is original max_len + 1
    # choose model type accordingly
    # prefer explicit model_type if present
    model_type_hint = ck_cfg.get("model_type", None)
    if model_type_hint is None:
        model_type = "nebula_cls" if ck_has_cls or (ck_max_len is not None and ck_max_len > (ck_cfg.get("max_len", 0) + 0)) else "nebula_tiny"
    else:
        model_type = model_type_hint

    # decide model max_len to build: if ck_max_len available and ck_has_cls -> set model max_len = ck_max_len - 1 for CLS
    if ck_max_len is not None:
        if model_type == "nebula_cls":
            model_max_len = max( ck_max_len - 1, int(ck_cfg.get("max_len", 256)) )
        else:
            model_max_len = int(ck_max_len)
    else:
        model_max_len = int(ck_cfg.get("max_len", 256))

    # build model
    if model_type == "nebula_cls" or ck_has_cls:
        model = NebulaCLS(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=model_max_len)
        print(f"[loader] Building NebulaCLS d_model={d_model} nhead={nhead} num_layers={num_layers} max_len={model_max_len}")
    else:
        model = NebulaTiny(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=model_max_len)
        print(f"[loader] Building NebulaTiny d_model={d_model} nhead={nhead} num_layers={num_layers} max_len={model_max_len}")

    # adapt positional encoding tensor in checkpoint if shapes mismatch
    # find pos key in checkpoint (common key 'pos.pe' in our model)
    pos_key = None
    for k in state_dict.keys():
        if k.endswith("pos.pe") or k.endswith("pe") or ".pe" in k:
            pos_key = k
            break

    if pos_key is not None and hasattr(model, "pos") and hasattr(model.pos, "pe"):
        ck_pe = state_dict[pos_key]  # tensor
        model_pe = model.pos.pe       # buffer
        ck_len = ck_pe.shape[1]
        model_len = model_pe.shape[1]
        if ck_len != model_len:
            print(f"[loader] pos.pe length mismatch: ck_len={ck_len}, model_len={model_len}. Adapting.")
            # case ck longer than model -> slice
            if ck_len > model_len:
                state_dict[pos_key] = ck_pe[:, :model_len, :].clone()
                print(f"[loader] Trimmed checkpoint pos.pe to model length {model_len}.")
            else:
                # ck shorter -> pad by repeating last vector
                pad_amt = model_len - ck_len
                last = ck_pe[:, -1:, :].clone()
                pads = last.repeat(1, pad_amt, 1)
                state_dict[pos_key] = torch.cat([ck_pe, pads], dim=1)
                print(f"[loader] Padded checkpoint pos.pe from {ck_len} -> {model_len} by repeating last vector.")
    else:
        # no pos key found or model lacks pos -> ignore
        pass

    # also handle 'cls_token' presence/mismatch:
    if "cls_token" in state_dict:
        # if model has cls_token param, ok; else remove it to avoid unexpected key
        if hasattr(model, "cls_token"):
            # if shapes mismatch, attempt to adapt
            ck_cls = state_dict.get("cls_token")
            model_cls_shape = tuple(getattr(model, "cls_token").shape)
            if tuple(ck_cls.shape) != model_cls_shape:
                # try to reshape/expand if possible
                try:
                    state_dict["cls_token"] = ck_cls.reshape(model_cls_shape)
                    print("[loader] reshaped cls_token from checkpoint to match model.")
                except Exception:
                    # fallback: remove checkpoint cls_token so load won't fail
                    print("[loader] removing checkpoint cls_token due to incompatible shape.")
                    del state_dict["cls_token"]
        else:
            # model has no cls_token -> remove it from ckpt so load_state_dict will not fail
            print("[loader] checkpoint has cls_token but current model does not. Removing key from state_dict.")
            del state_dict["cls_token"]

    # strip keys that do not exist in model (avoid unexpected keys)
    model_keys = set(model.state_dict().keys())
    cleaned_state = {}
    for k, v in state_dict.items():
        if k in model_keys:
            cleaned_state[k] = v
        else:
            # attempt to remove 'module.' prefix
            if k.startswith("module.") and k[len("module."): ] in model_keys:
                cleaned_state[k[len("module."):]] = v
            else:
                # skip unexpected key quietly
                print(f"[loader] skipping unexpected ckpt key: {k}")

    # finally load tolerant
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    # PyTorch returns namedtuple with missing_keys/unexpected_keys only in error; print diagnostics if possible
    print("[loader] load_state_dict completed (missing or mismatched keys may remain).")
    return model

def softmax_probs(logits):
    # logits: [B,2] -> probs for class 1
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
    return probs

# -------------------------
# Core functions
# -------------------------
def generate_windows_from_ids(ids, events_per_window, stride):
    """
    ids: list[int] (padded/truncated to maxlen already)
    We'll create windows over the sequence of token ids.
    Each window is a contiguous slice of length events_per_window.
    Return: list of (start_idx, window_ids_list)
    """
    L = len(ids)
    windows = []
    if events_per_window <= 0 or events_per_window > L:
        # single window: whole sequence
        windows.append((0, ids[:]))
        return windows
    for start in range(0, L - events_per_window + 1, stride):
        w = ids[start:start+events_per_window]
        windows.append((start, w))
    # if last window not aligned and we want to include tail, include final window
    if len(windows) == 0:
        windows.append((0, ids[:events_per_window]))
    return windows

def batch_infer(model, device, batch_ids):
    """
    batch_ids: torch.LongTensor [B, T]
    returns: numpy array of shape [B] of P(malware)
    """
    model.eval()
    with torch.no_grad():
        logits = model(batch_ids.to(device))
        probs = F.softmax(logits, dim=-1)[:,1].cpu().numpy()
    return probs

# -------------------------
# Main script
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="TSV: sid\\tlabel\\ttrace")
    ap.add_argument("--ckpt", required=True, help="model checkpoint path (.pt)")
    ap.add_argument("--vocab", required=True, help="vocab json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--window_unit", choices=["event","token"], default="event")
    ap.add_argument("--force_cpu", action="store_true")
    # sample-level rules
    ap.add_argument("--thr", type=float, default=None, help="threshold for P(malware) to mark window as malware")
    ap.add_argument("--zero_fp", type=str, default=None, help="json file with {'thr':...} to load thr")
    ap.add_argument("--minwins", type=int, default=0, help="require at least this many malware windows to mark sample malware")
    ap.add_argument("--consecutive_k", type=int, default=0, help="require k consecutive malware windows")
    ap.add_argument("--split_dirs", action="store_true", help="save per-sample PNGs into classes folder")
    ap.add_argument("--save_score_hist", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load threshold if provided via JSON
    thr = args.thr
    if args.zero_fp:
        try:
            z = json.load(open(args.zero_fp, "r", encoding="utf-8"))
            for k in ("thr","threshold","value"):
                if k in z:
                    thr = float(z[k])
                    break
        except Exception as e:
            print("Warning: could not read zero_fp json:", e)

    # device
    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    # tokenizer
    tok = Tokenizer(args.vocab)

    # load checkpoint
    state_dict, ck_cfg = load_checkpoint(args.ckpt, device)
    # merge config if model config absent
    if not isinstance(ck_cfg, dict):
        ck_cfg = {}

    # infer vocab size
    try:
        vocab_json = json.load(open(args.vocab, "r", encoding="utf-8"))
        vocab_size = len(vocab_json)
    except Exception:
        vocab_size = ck_cfg.get("vocab_size", 25000)

    model = robust_build_and_load_model(state_dict, ck_cfg, vocab_size, device)
    model.to(device)
    model.eval()
    # model = build_model_from_ck(ck_cfg, vocab_size, device)
    # load weights (expecting 'state_dict' compatible)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        # try tolerant load
        print("State dict load error:", e)
        # attempt to remove 'module.' prefixes
        new_state = {}
        for k,v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[len("module."):]
            new_state[name] = v
        model.load_state_dict(new_state, strict=False)

    # read data file (TSV: sid, label, trace)
    samples = []
    with open(args.data_file, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t", 2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0].strip(), int(parts[1]), parts[2]
            samples.append((sid, lab, trace))
    print("Loaded samples:", len(samples))

    # build windows list (global) -> we want to process in batches
    # keep mapping sample -> list of (start_idx, window_ids)
    sample_windows = {}
    total_windows = 0
    for sid, lab, trace in samples:
        ids = tok.encode(trace, max_len=args.max_len)  # padded/truncated
        wins = generate_windows_from_ids(ids, args.events_per_window, args.stride_events)
        sample_windows[sid] = {"label": lab, "windows": wins}
        total_windows += len(wins)

    print(f"Total windows to score: {total_windows}")

    # iterate windows in batches
    all_scores_by_sid = defaultdict(list)
    B = args.batch_size
    # create iterator of (sid, start, win_ids)
    iter_list = []
    for sid, info in sample_windows.items():
        for start, w in info["windows"]:
            iter_list.append((sid, start, w))
    # process in batches
    t0 = time.time()
    for i in tqdm(range(0, len(iter_list), B), desc="Scoring windows"):
        batch = iter_list[i:i+B]
        batch_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
        probs = batch_infer(model, device, batch_ids)
        for (sid, start, w), p in zip(batch, probs):
            all_scores_by_sid[sid].append(float(p))
    t1 = time.time()
    print(f"Scoring done in {t1-t0:.1f}s")

    # write window_eval.json
    out_json = {"meta": {"data_file": args.data_file, "ckpt": args.ckpt, "vocab": args.vocab,
                         "cfg": ck_cfg},
                "samples": {}}
    for sid, info in sample_windows.items():
        lab = info["label"]
        scores = all_scores_by_sid.get(sid, [])
        out_json["samples"][sid] = {"label": lab, "n_windows": len(scores), "scores": scores}
    json_path = outdir / "window_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f)
    print("Wrote", json_path)

    # score histogram
    if args.save_score_hist:
        flat = [s for sid in out_json["samples"] for s in out_json["samples"][sid]["scores"]]
        if len(flat) > 0:
            plt.figure(figsize=(6,3))
            plt.hist(flat, bins=200)
            plt.title("Window score histogram")
            plt.xlabel("P(malware)")
            plt.tight_layout()
            histp = outdir / "window_eval.score_hist.png"
            plt.savefig(histp, dpi=150)
            plt.close()
            print("Wrote", histp)

    # optional: apply sample-level decision rule and save per-class folders + CSV summary
    if thr is not None:
        print("Applying sample-level rule with thr=", thr, " minwins=", args.minwins, " consecutive_k=", args.consecutive_k)
        # helper funcs
        def sample_pred_from_scores(scores, thr, minwins=0, consecutive_k=0):
            cls = [1 if s >= thr else 0 for s in scores]
            if consecutive_k and consecutive_k > 0:
                # check for run length >= consecutive_k
                run = 0
                for v in cls:
                    if v == 1:
                        run += 1
                        if run >= consecutive_k:
                            return 1
                    else:
                        run = 0
                return 0
            if minwins and minwins > 0:
                return 1 if sum(cls) >= minwins else 0
            # default OR rule
            return 1 if any(cls) else 0

        # create dirs if requested
        if args.split_dirs:
            split_root = outdir / "classes_split"
            malware_before = split_root / "malware" / "before"
            good_before = split_root / "goodware" / "before"
            malware_before.mkdir(parents=True, exist_ok=True)
            good_before.mkdir(parents=True, exist_ok=True)

        # CSV summary
        csvp = outdir / "sample_level_from_windows.csv"
        with open(csvp, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["sid", "label", "n_windows", "mal_windows", "sample_pred"])
            for sid, entry in out_json["samples"].items():
                lab = int(entry["label"])
                scores = entry["scores"]
                mal_windows = sum(1 for s in scores if s >= thr)
                sample_pred = sample_pred_from_scores(scores, thr, args.minwins, args.consecutive_k)
                writer.writerow([sid, lab, len(scores), mal_windows, sample_pred])
                # optional save small class timeline PNGs if split_dirs
                if args.split_dirs:
                    # simple step plot
                    cls = [1 if s >= thr else 0 for s in scores]
                    plt.figure(figsize=(6,1.8))
                    plt.step(range(len(cls)), cls, where="mid", linewidth=1.2)
                    plt.ylim(-0.1,1.1); plt.yticks([0,1], ["goodware","malware"])
                    plt.title(f"SID {sid} label={lab} pred={sample_pred} mal_wins={mal_windows}")
                    target_dir = malware_before if sample_pred == 1 else good_before
                    target_dir.mkdir(parents=True, exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(target_dir / f"sample_{sid}.png", dpi=120)
                    plt.close()
        print("Wrote sample-level CSV:", csvp)

    print("Done.")

if __name__ == "__main__":
    main()
