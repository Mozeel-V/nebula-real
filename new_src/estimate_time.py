#!/usr/bin/env python3
"""
Estimates runtime of the strong saliency attack:
 - Measures forward pass cost
 - Measures saliency (gradient) cost (via embeddings)
 - Measures per-candidate test cost
 - Predicts time per iteration, per sample, and total time for full dataset.

Usage example:
 python new_src/estimate_time.py \
    --ckpt checkpoints/run_weighted_w0_8/best.pt \
    --vocab checkpoints/vocab_n.json \
    --in_tsv data/dataset_small_2k_normalized.tsv \
    --api_candidates checkpoints/500_api_candidates.json \
    --max_len 256 \
    --cand_sample 200 \
    --iter_steps 3 \
    --n_replace_total 12 \
    --topk_salient 20
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# adjust imports to your repo layout
from tokenizer import Tokenizer
from nebula_model import NebulaCLS, NebulaTiny

# -------------------------
# model loader (best-effort)
# -------------------------
def load_model(ckpt_path, vocab_size, device):
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        cfg = ck.get("config", {})
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
        cfg = ck.get("config", {})
    else:
        state = ck
        cfg = {}

    has_cls = any(k.endswith("cls_token") or k == "cls_token" for k in state.keys())
    d_model = int(cfg.get("d_model", 128))
    nhead = int(cfg.get("nhead", 4))
    num_layers = int(cfg.get("num_layers", cfg.get("layers", 2)))
    ff = int(cfg.get("ff", cfg.get("dim_feedforward", 256)))
    max_len = int(cfg.get("max_len", 256))

    if has_cls or cfg.get("model_type") == "nebula_cls" or num_layers >= 3:
        model = NebulaCLS(vocab_size, d_model=d_model, nhead=nhead,
                          num_layers=num_layers, dim_feedforward=ff,
                          max_len=max_len)
    else:
        model = NebulaTiny(vocab_size, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, dim_feedforward=ff,
                           max_len=max_len)
    # tolerant load
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        new_state = {}
        for k, v in state.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Timing utilities
# -------------------------
def timed_forward(model, ids, device, repeats=5):
    xs = torch.tensor([ids], dtype=torch.long, device=device)
    # warmup
    for _ in range(3):
        _ = model(xs)
    t0 = time.time()
    for _ in range(repeats):
        _ = model(xs)
    t1 = time.time()
    return (t1 - t0) / repeats

def timed_saliency(model, ids, device, repeats=3):
    """
    Compute time to get gradient-based saliency using embeddings.
    Uses model.forward_from_embeddings if present.
    """
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    emb_layer = model.embed
    # warmup
    for _ in range(2):
        emb = emb_layer(ids_tensor)
        emb2 = emb.detach().clone()
        emb2.requires_grad_()
        if hasattr(model, "forward_from_embeddings"):
            logits = model.forward_from_embeddings(emb2)
        else:
            logits = model(ids_tensor)
        loss = logits[:, 1].sum()
        loss.backward()
    # timed
    t0 = time.time()
    for _ in range(repeats):
        emb = emb_layer(ids_tensor)
        emb2 = emb.detach().clone()
        emb2.requires_grad_()
        if hasattr(model, "forward_from_embeddings"):
            logits = model.forward_from_embeddings(emb2)
        else:
            logits = model(ids_tensor)
        loss = logits[:, 1].sum()
        loss.backward()
    t1 = time.time()
    return (t1 - t0) / repeats

def timed_candidate_test(model, ids, candidate_ids, device, repeats=20):
    xs = torch.tensor([ids], dtype=torch.long, device=device)
    L = len(ids)
    pos = np.random.randint(0, max(1, L))
    orig = xs.clone()
    # warmup
    for _ in range(3):
        xs2 = orig.clone()
        xs2[0, pos] = candidate_ids[0]
        _ = model(xs2)
    t0 = time.time()
    for i in range(repeats):
        xs2 = orig.clone()
        xs2[0, pos] = candidate_ids[i % len(candidate_ids)]
        _ = model(xs2)
    t1 = time.time()
    return (t1 - t0) / repeats

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--api_candidates", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--cand_sample", type=int, default=200)
    ap.add_argument("--iter_steps", type=int, default=3)
    ap.add_argument("--n_replace_total", type=int, default=12)
    ap.add_argument("--topk_salient", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # device fixed to cpu per your environment choice
    device = torch.device("cpu")
    print("Device:", device)

    # robust vocab loader (list or dict)
    vocab_raw = json.load(open(args.vocab, "r", encoding="utf-8"))
    if isinstance(vocab_raw, list):
        vocab_dict = {tok: idx for idx, tok in enumerate(vocab_raw)}
    elif isinstance(vocab_raw, dict):
        try:
            vocab_dict = {k: int(v) for k, v in vocab_raw.items()}
        except Exception:
            vocab_dict = vocab_raw
    else:
        vocab_dict = {}
    unk_id = vocab_dict.get("<unk>", 1)

    # tokenizer
    tok = Tokenizer(args.vocab)

    # load model
    vocab_size = len(vocab_dict) if vocab_dict else 25000
    model = load_model(args.ckpt, vocab_size, device)
    print("[loaded model]")

    # prepare a small subset (3 samples) for timing
    rows = []
    with open(args.in_tsv, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            rows.append(parts)
            if len(rows) >= 3:
                break
    if not rows:
        raise RuntimeError("No rows found in in_tsv")

    sid, lab, trace = rows[0]
    ids = tok.encode(trace, max_len=args.max_len)

    # measure forward
    fwd = timed_forward(model, ids, device)
    print(f"[forward pass] avg {fwd:.6f} sec")

    # measure saliency
    sal = timed_saliency(model, ids, device)
    print(f"[saliency pass] avg {sal:.6f} sec")

    # load api candidates and prepare candidate ids
    api_cands = json.load(open(args.api_candidates, "r", encoding="utf-8"))
    if not isinstance(api_cands, list):
        # if candidates json was saved as dict, try to extract keys or values
        if isinstance(api_cands, dict):
            api_cands = list(api_cands.keys())
        else:
            api_cands = list(api_cands)

    cand_ids = [vocab_dict.get(tokstr, unk_id) for tokstr in api_cands if isinstance(tokstr, str)]
    if len(cand_ids) == 0:
        # fallback to some random tokens within vocab range
        cand_ids = list(range(min(500, vocab_size)))
    random.shuffle(cand_ids)
    candidate_ids = cand_ids[:min(20, len(cand_ids))]

    cand_t = timed_candidate_test(model, ids, candidate_ids, device)
    print(f"[candidate test] avg {cand_t:.6f} sec per candidate")

    # compute estimated times
    topk = int(args.topk_salient)
    cand_sample = int(args.cand_sample)
    iter_steps = int(args.iter_steps)

    # per-iteration time estimate:
    # saliency + for topk positions try cand_sample candidates each (cand_sample * candidate_time)
    iter_time = sal + topk * cand_sample * cand_t
    sample_time = iter_steps * iter_time

    # count malware samples in dataset
    malware_count = 0
    with open(args.in_tsv, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t", 2)
            if len(parts) < 3: continue
            if int(parts[1]) == 1:
                malware_count += 1

    total_time = malware_count * sample_time

    print("\n=== ESTIMATE ===")
    print(f"malware samples: {malware_count}")
    print(f"topk_salient: {topk}")
    print(f"time per iteration (saliency + tests): {iter_time:.3f} sec")
    print(f"time per sample (iter_steps={iter_steps}): {sample_time:.3f} sec")
    print(f"TOTAL estimated: ~{total_time/60:.2f} minutes (~{total_time/3600:.2f} hours)")
    print("================")
    # recommendation hint
    limit_min = 25
    if total_time/60 > limit_min:
        print(f"[hint] Estimated total > {limit_min} min. To reduce time, lower cand_sample, topk_salient, or iter_steps.")
    else:
        print(f"[hint] Estimated total within {limit_min} min target.")

if __name__ == "__main__":
    main()
