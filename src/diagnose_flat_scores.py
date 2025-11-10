#!/usr/bin/env python3
"""
Diagnostic: find why model outputs identical probabilities.

Usage:
  python debug_scripts/diagnose_flat_scores.py \
    --ckpt checkpoints/best.pt \
    --vocab checkpoints/vocab.json \
    --sample_tsv dataset_small_2k.tsv \
    --n_examples 6

It prints:
 - tokenization/token-id variety for sample lines
 - vocab size
 - checkpoint keys and missing keys when loading model
 - embedding weight stats (mean/std/min/max)
 - model logits/probs on a handful of diverse inputs
 - uniques of window scores across dataset
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

from tokenizer import tokenize, load_vocab, tokens_to_ids
from nebula_model import NebulaTiny

def try_load_ckpt(ckpt_path, vocab_path=None, device=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint keys:", list(ck.keys()))
    cfg = ck.get("config") or ck.get("cfg") or {}
    print("Config found in checkpoint:", bool(bool(cfg)))
    # infer model params
    vocab = None
    if vocab_path:
        try:
            vocab = load_vocab(vocab_path)
        except Exception as e:
            print("Failed to load vocab:", e)
    vocab_size = cfg.get("vocab_size", len(vocab) if vocab else None)
    print("Inferred vocab_size:", vocab_size)
    try:
        d_model = cfg.get("d_model", 128)
        nhead = cfg.get("nhead", cfg.get("heads", 4))
        num_layers = cfg.get("num_layers", cfg.get("layers", 2))
        ff = cfg.get("ff", cfg.get("dim_feedforward", 256))
        max_len = cfg.get("max_len", 512)
        num_classes = cfg.get("num_classes", 2)
    except Exception:
        d_model, nhead, num_layers, ff, max_len, num_classes = 128,4,2,256,512,2

    model = NebulaTiny(
        vocab_size=(vocab_size or 30000),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff,
        max_len=max_len,
        num_classes=num_classes
    )
    # pick likely state dict key
    state = ck.get("model", ck.get("model_state", ck.get("model_state_dict", ck.get("state_dict", ck))))
    try:
        miss = model.load_state_dict(state, strict=False)
        print("load_state_dict() returned:", miss)
    except Exception as e:
        print("Error loading state_dict:", e)
    return model, cfg

def inspect_vocab_and_tokenization(vocab_path, tsv_path, n_examples=6):
    print("\n=== Vocab + Tokenization check ===")
    vocab = load_vocab(vocab_path)
    print("Vocab size:", len(vocab))
    # sample lines
    samples = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln=ln.rstrip("\n")
            if not ln: continue
            sid, lab, trace = ln.split("\t",2)
            samples.append((sid, lab, trace))
            if len(samples)>=n_examples: break
    for sid, lab, trace in samples:
        toks = tokenize(trace)
        ids = tokens_to_ids(toks, vocab, max_len=128)
        uniq_ids = len(set(ids))
        print(f"SID={sid} label={lab} tokens={len(toks)} unique_token_ids_in_padded_seq={uniq_ids} first10_ids={ids[:10]} first10_toks={toks[:10]}")

def embedding_stats(model):
    print("\n=== Embedding stats ===")
    try:
        emb_w = model.get_embedding_weight() if hasattr(model, "get_embedding_weight") else model.embed.weight.data
        arr = emb_w.cpu().numpy()
        print("embedding shape:", arr.shape)
        print("emb mean %.6f std %.6f min %.6f max %.6f unique_vals:%d" %
              (float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), int(len(np.unique(np.round(arr.flatten(),6))))))
    except Exception as e:
        print("Could not inspect embeddings:", e)

def model_sensitivity_test(model, vocab_path):
    print("\n=== Model sensitivity test ===")
    vocab = load_vocab(vocab_path)
    texts = [
        "api:ReadFile path:C:\\\\Windows\\\\Temp\\\\pad0.tmp ||| api:CreateFileW path:C:\\\\Temp\\\\a.txt",
        "api:connect ip:8.8.8.8 ||| api:send ip:8.8.8.8",
        "random gibberish noapi token token token",
        "api:RegOpenKeyExW path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor",
        "api:CreateProcess pid:1234 ||| api:WriteFile path:C:\\\\Temp\\\\x.dll"
    ]
    model.eval()
    device = next(model.parameters()).device
    for t in texts:
        toks = tokenize(t)
        ids = tokens_to_ids(toks, vocab, max_len=128)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        print("text:", t[:80], "... -> logits:", [float(round(x,6)) for x in logits[0].cpu().numpy().tolist()], "probs:", [round(p,6) for p in probs])

def window_score_uniques(window_eval_json_path, n_check=50):
    print("\n=== Window score uniqueness check (first samples) ===")
    j = json.load(open(window_eval_json_path, "r", encoding="utf-8"))
    samples = j.get("samples", {})
    cnt=0
    for sid, info in samples.items():
        sc = info.get("scores", [])
        if not sc:
            print("sid", sid, "has no scores")
        else:
            uniq = len(set([round(x,6) for x in sc]))
            print("sid", sid, "nwin", len(sc), "unique_scores", uniq, "max", max(sc))
        cnt+=1
        if cnt>=n_check: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--sample_tsv", required=True)
    ap.add_argument("--window_eval", default=None, help="path to existing window_eval.json to check score uniqueness")
    ap.add_argument("--n_examples", type=int, default=6)
    args = ap.parse_args()

    # vocab + tokenization
    inspect_vocab_and_tokenization(args.vocab, args.sample_tsv, args.n_examples)

    # model + loading
    model, cfg = try_load_ckpt(args.ckpt, args.vocab)
    # move model to cpu and eval
    model.to(torch.device("cpu"))
    model.eval()

    # embedding stats
    embedding_stats(model)

    # model sensitivity
    model_sensitivity_test(model, args.vocab)

    # window_eval check if provided
    if args.window_eval:
        window_score_uniques(args.window_eval, n_check=20)

if __name__ == "__main__":
    import argparse, json
    main()
