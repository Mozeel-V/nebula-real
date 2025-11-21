#!/usr/bin/env python3
"""
Estimate per-API effectiveness by replacing high-saliency token positions
with generated API events and measuring the drop in P(malware).

Outputs: JSON list [{"api": "api:ReadFile", "mean_delta": 0.12, "count": 50}, ...]

Usage: python compute_api_effectiveness.py \
  --dataset dataset_small_2k.tsv \
  --api_pool checkpoints/200_api_candidates.json \
  --saliency_json results/saliency_positions.json \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out checkpoints/api_effectiveness.json \
  --sample_limit 500 --k 32 --per_api_positions 2

"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
import torch

from old_tokenizer import tokenize, load_vocab, tokens_to_ids
from nebula_model import NebulaTiny

# -------------------------
# Model loader helper
# -------------------------
def load_model_ckpt(ckpt_path, vocab_path=None, device=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config") or ck.get("cfg") or {}
    vocab = None
    if vocab_path:
        vocab = load_vocab(vocab_path)
    vocab_size = cfg.get("vocab_size", len(vocab) if vocab else 30000)
    d_model = cfg.get("d_model", 128)
    nhead = cfg.get("nhead", cfg.get("heads", 4))
    num_layers = cfg.get("num_layers", cfg.get("layers", 2))
    ff = cfg.get("ff", cfg.get("dim_feedforward", 256))
    max_len = cfg.get("max_len", 512)
    num_classes = cfg.get("num_classes", 2)
    chunk_size = cfg.get("chunk_size", 0)

    model = NebulaTiny(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff,
        max_len=max_len,
        num_classes=num_classes,
        chunk_size=chunk_size
    )

    state = ck.get("model", ck.get("model_state", ck.get("state_dict", ck)))
    if state is None:
        raise RuntimeError(f"No model weights found in checkpoint {ckpt_path}; keys: {list(ck.keys())}")
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print("[WARN] Missing keys when loading state_dict:", missing.missing_keys)
    if missing.unexpected_keys:
        print("[WARN] Unexpected keys:", missing.unexpected_keys)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device).eval()
    return model, cfg

# -------------------------
# Utilities
# -------------------------
def read_tsv_dataset(tsv_path: str, only_malware: bool = True) -> List[tuple]:
    samples = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            sid, lab, trace = ln.split("\t", 2)
            lab = int(lab)
            if only_malware and lab != 1:
                continue
            samples.append((sid, lab, trace))
    return samples

def sample_event_from_api(api_token: str, idx: int = 0) -> str:
    """
    Small rule-based generator for an API event string from an api token.
    Keep event realistic but simple.
    """
    api = api_token.split("api:")[-1]
    low = api.lower()
    if "readfile" in low or "writefile" in low or "createfile" in low:
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
    if "connect" in low or "send" in low or "recv" in low:
        return f"api:{api} ip:127.0.0.1"
    if low.startswith("reg"):
        return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
    if "process" in low:
        return f"api:{api} pid:{1000 + idx}"
    return f"api:{api}"

def replace_token_at_index(flat_trace: str, token_idx: int, new_event: str) -> str:
    """
    Replace the token at index `token_idx` in a flattened trace.
    Supports event-separated traces using " ||| " and fallback whitespace split.
    """
    if " ||| " in flat_trace:
        parts = flat_trace.split(" ||| ")
        if 0 <= token_idx < len(parts):
            parts[token_idx] = new_event
        return " ||| ".join(parts)
    else:
        parts = flat_trace.split()
        if 0 <= token_idx < len(parts):
            parts[token_idx] = new_event
        return " ".join(parts)

# score single flattened trace and return P(malware)
class ModelScorer:
    def __init__(self, ckpt_path, vocab_path, device=None):
        self.model, self.cfg = load_model_ckpt(ckpt_path, vocab_path, device=device)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.vocab = load_vocab(vocab_path)

    def score_trace(self, flat_trace: str, max_len: int = 0) -> float:
        max_len = max_len or int(self.cfg.get("max_len", 256))
        toks = tokenize(flat_trace)
        ids = tokens_to_ids(toks, self.vocab, max_len=max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)  # [1,2]
            probs = torch.softmax(logits, dim=-1)[0,1].item()
        return float(probs)

# Use the tokenizer helper (we re-import here to ensure local scope)
from old_tokenizer import tokens_to_ids, load_vocab

# -------------------------
# Main estimation routine
# -------------------------
def estimate_api_effectiveness(args):
    # load api pool
    api_pool = json.load(open(args.api_pool, "r", encoding="utf-8"))
    print(f"Loaded {len(api_pool)} API candidates")

    # load saliency map (sid -> list of token indices)
    saliency = json.load(open(args.saliency_json, "r", encoding="utf-8"))

    # load dataset (malware only)
    samples = read_tsv_dataset(args.dataset, only_malware=True)
    if args.sample_limit and args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    print(f"Using {len(samples)} malware samples for estimation")

    # scorer
    scorer = ModelScorer(args.ckpt, args.vocab, device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

    api_stats = defaultdict(list)
    for sid, lab, trace in samples:
        try:
            orig_score = scorer.score_trace(trace)
        except Exception as e:
            print(f"[WARN] scoring original trace failed for sid={sid}: {e}")
            continue

        # get saliency positions for this sid
        pos_list = saliency.get(str(sid), saliency.get(int(sid), []))
        if not pos_list:
            # fallback: try split into tokens/events and pick first few indices
            parts = trace.split(" ||| ") if " ||| " in trace else trace.split()
            pos_list = list(range(min(len(parts), args.k)))
        # constrain positions
        pos_list = pos_list[: args.k]

        for api in api_pool:
            # sample positions to try (limit per api)
            positions = random.sample(pos_list, min(len(pos_list), args.per_api_positions))
            for i, pos in enumerate(positions):
                new_event = sample_event_from_api(api, idx=random.randint(0, 9999))
                adv = replace_token_at_index(trace, pos, new_event)
                try:
                    new_score = scorer.score_trace(adv)
                except Exception as e:
                    # skip if scoring fails
                    continue
                delta = float(orig_score) - float(new_score)
                api_stats[api].append(delta)

    # compute mean delta per api
    out = []
    for api, deltas in api_stats.items():
        if len(deltas) == 0:
            mean = 0.0
        else:
            mean = float(np.mean(deltas))
        out.append({"api": api, "mean_delta": mean, "count": len(deltas)})
    out_sorted = sorted(out, key=lambda x: x["mean_delta"], reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_sorted, f, indent=2)
    print(f"Wrote api effectiveness -> {args.out}")

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="TSV dataset (id \\t label \\t trace)")
    p.add_argument("--api_pool", required=True, help="JSON list of api tokens (checkpoints/api_candidates.json)")
    p.add_argument("--saliency_json", required=True, help="saliency_positions.json")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", default="checkpoints/api_effectiveness.json")
    p.add_argument("--sample_limit", type=int, default=200, help="max malware samples to use (0 => all)")
    p.add_argument("--k", type=int, default=32, help="top-k saliency positions to consider per sample")
    p.add_argument("--per_api_positions", type=int, default=2, help="positions to try per api per sample")
    args = p.parse_args()
    estimate_api_effectiveness(args)

if __name__ == "__main__":
    main()
