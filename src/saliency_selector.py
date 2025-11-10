#!/usr/bin/env python3
"""
Compute token-level saliency (gradient-based) for traces in a TSV dataset.

Outputs: JSON map { sid: [top_k_token_indices,...], ... }
Format of TSV expected: sid\\tlabel\\ttrace

Algorithm (per sample):
  - tokenize trace -> ids (max_len)
  - get embedding tensor for ids (requires_grad)
  - forward_from_embeddings(emb) -> logits
  - compute gradient of malware logit (class 1) wrt embeddings
  - per-token saliency = L2 norm of gradient vector for that token
  - pick top_k indices

Notes:
 - Uses CPU by default (device autodetects CUDA if available)
 - sample_limit controls how many malware samples to process (0 => all)
 - This is a simple, effective saliency used for targeted attacks

Usage: python saliency_selector.py --data_file dataset_small_2k.tsv --ckpt checkpoints/best.pt --vocab checkpoints/vocab.json --device cpu --process_only_malware --sample_limit 500
"""
import argparse
import json
import math
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

from tokenizer import tokenize, load_vocab, tokens_to_ids
from nebula_model import NebulaTiny

# Helper: load model like window_eval_plot did
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
    # non-strict load is fine for small differences
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print("[WARN] Missing keys when loading state_dict:", missing.missing_keys)
    if missing.unexpected_keys:
        print("[WARN] Unexpected keys:", missing.unexpected_keys)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device).eval()
    return model, cfg

def compute_saliency_for_ids(model, ids, device, max_len):
    """
    ids: list[int] length <= max_len
    returns: saliency list length == len(ids) (float)
    """
    # prepare tensor [1, T]
    T = len(ids)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    # get embeddings tensor using embed layer (B,T,D)
    emb = model.embed(inp)         # [1, T, D]
    emb = emb.detach()             # ensure we start fresh
    emb.requires_grad_()           # need grads on embeddings
    # forward_from_embeddings expects embeddings and returns logits
    logits = model.forward_from_embeddings(emb)  # [1, num_classes]
    # score for malware class (index 1)
    score = logits[0, 1]
    # backward to get grad wrt embeddings
    model.zero_grad()
    score.backward(retain_graph=False)
    grad = emb.grad  # [1, T, D]
    if grad is None:
        # fallback: zero saliency
        return [0.0] * T
    # compute L2 norm per token
    sal = torch.norm(grad[0], dim=1).cpu().numpy().tolist()  # length T
    return sal

def topk_indices_from_saliency(saliency, k):
    if not saliency:
        return []
    k = min(k, len(saliency))
    arr = np.array(saliency)
    # argsort descending
    idx = np.argsort(-arr)[:k]
    return idx.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="TSV dataset (sid\\tlabel\\ttrace)")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint (best.pt)")
    ap.add_argument("--vocab", required=True, help="vocab json (list)")
    ap.add_argument("--out", default="results/saliency_positions.json")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=64, help="top-k token indices to store per sample")
    ap.add_argument("--sample_limit", type=int, default=200, help="number of malware samples to process (0 => all)")
    ap.add_argument("--process_only_malware", action="store_true", help="compute saliency only for samples with label==1 (malware)")
    ap.add_argument("--device", default=None, help="device to run on, e.g., cpu or cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)

    model, cfg = load_model_ckpt(args.ckpt, args.vocab, device=device)
    vocab = load_vocab(args.vocab)

    # load samples
    samples = []
    with open(args.data_file, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln: continue
            sid, lab, trace = ln.split("\t", 2)
            samples.append((sid, int(lab), trace))
    print("Loaded", len(samples), "samples from", args.data_file)

    out = {}
    processed = 0
    to_process = [s for s in samples if (not args.process_only_malware) or s[1] == 1]
    if args.sample_limit and args.sample_limit > 0:
        to_process = to_process[: args.sample_limit]
    print("Will compute saliency for", len(to_process), "samples (process_only_malware=%s)" % args.process_only_malware)

    for sid, lab, trace in tqdm(to_process):
        # tokenize into tokens (preserve event separator rules are not needed here)
        toks = tokenize(trace)
        # convert to ids (no padding truncation inside tokens_to_ids but we need exact length)
        ids_full = [tokens_to_ids(toks, vocab, max_len=args.max_len)]
        # tokens_to_ids returns padded list length max_len; find effective tokens where not PAD
        # but simpler: re-tokenize and map tokens until PAD token encountered
        # we need actual token length before padding:
        toks_truncated = toks[: args.max_len]
        ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in toks_truncated]
        if len(ids) == 0:
            # empty trace -> skip or mark empty
            out[sid] = []
            processed += 1
            continue
        # compute saliency
        try:
            sal = compute_saliency_for_ids(model, ids, device, args.max_len)
        except Exception as e:
            print(f"[WARN] saliency failed for sid {sid}: {e}")
            out[sid] = []
            processed += 1
            continue
        topk = topk_indices_from_saliency(sal, args.top_k)
        out[sid] = topk
        processed += 1
        # Light checkpointing to disk every 100 processed samples
        if processed % 100 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    # final write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote saliency positions to", args.out)
    print("Done.")

if __name__ == "__main__":
    main()
