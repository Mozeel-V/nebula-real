#!/usr/bin/env python3
"""
Saliency-guided iterative in-vocab replacement attack (greedy).

Saves attacked dataset as TSV: sid \t label \t trace

Notes:
 - Requires: src/tokenizer.py, src/models/nebula_model.py (NebulaCLS or NebulaTiny)
 - Requires in-vocab candidates JSON produced earlier (results/api_candidates_in_vocab.json)
 - Works on whole trace but computes saliency per-token by backprop through embeddings.
 - Defaults tuned for small 2k dataset and CPU. Increase `n_replace_total` and `cand_sample` for stronger attack if you have more time.

Usage example:
python src/attacks/saliency_attack_strong.py \
  --in_tsv data/dataset_small_2k_normalized.tsv \
  --ckpt checkpoints/run_weighted_w0_8/best.pt \
  --vocab checkpoints/vocab_n.json \
  --api_candidates results/api_candidates_in_vocab.json \
  --out results/attacks/saliency_strong_2k.tsv \
  --only_malware \
  --n_replace_total 12 \
  --topk_salient 20 \
  --cand_sample 200 \
  --iter_steps 3 \
  --batch_size 64

"""
import argparse, json, random, math, time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from tokenizer import Tokenizer
from nebula_model import NebulaTiny, NebulaCLS

# ---------- helper: load checkpoint and build model (simple, assumes compatibility) ----------
def load_model_simple(ckpt_path, vocab_size, device):
    ck = torch.load(ckpt_path, map_location=device)
    # support checkpoints saved as {"model": state_dict, "config": {...}}
    state = None
    cfg = {}
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        cfg = ck.get("config", {})
    else:
        state = ck
    # choose model variant: if state has 'cls_token' use NebulaCLS
    has_cls = any(k.endswith("cls_token") or k == "cls_token" for k in state.keys())
    d_model = int(cfg.get("d_model", 128))
    nhead = int(cfg.get("nhead", 4))
    num_layers = int(cfg.get("num_layers", 2))
    ff = int(cfg.get("ff", cfg.get("dim_feedforward", 256)))
    max_len = int(cfg.get("max_len", 256))
    if has_cls or cfg.get("model_type") == "nebula_cls" or num_layers >= 3:
        model = NebulaCLS(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=max_len)
    else:
        model = NebulaTiny(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=ff, max_len=max_len)
    # try to load tolerant
    try:
        model.load_state_dict(state)
    except Exception:
        # try strip module. and adapt pe if necessary (best effort)
        new_state = {}
        for k,v in state.items():
            nk = k[len("module."): ] if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model

# ---------- utilities ----------
def tokenize_and_ids(tok, trace, max_len):
    ids = tok.encode(trace, max_len=max_len)
    return ids

def ids_to_trace_from_tokens(tok, ids):
    # produce a trace string from token ids (used for saving)
    toks = tok.decode(ids)
    # join preserving token spacing
    return " ".join(toks)

# ---------- saliency functions ----------
def compute_token_saliency_for_sequence(model, device, input_ids, target_class=1):
    """
    input_ids: list[int] length L
    Returns saliency per token (L,) — absolute gradient norm of the malware logit w.r.t token embedding
    Implementation:
     - convert input_ids -> embedding tensor via model.embed (embedding lookup)
     - set requires_grad on embeddings
     - forward through model.forward_from_embeddings (if available) or model(ids) with embeddings substitution
    """
    model.eval()
    # prepare tensor
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1,L]
    # get embedding layer
    embed = model.get_embedding_weight()  # returns weight tensor on cpu maybe
    # We will use model.embed so we'll do forward with embeddings requiring grads
    emb_layer = model.embed
    # lookup embeddings
    emb = emb_layer(ids_tensor)  # [1,L,D]
    emb = emb.detach().clone()
    emb.requires_grad_()
    # forward using forward_from_embeddings if present
    if hasattr(model, "forward_from_embeddings"):
        logits = model.forward_from_embeddings(emb)  # [1,2]
    else:
        # fallback: we do forward by mapping emb to model forward - but models accept ids. Not ideal.
        logits = model(ids_tensor)
    # pick malware logit (assuming logits raw)
    malware_logit = logits[:, target_class].sum()
    malware_logit.backward(retain_graph=False)
    grads = emb.grad  # [1,L,D]
    saliency = grads.abs().sum(dim=-1).squeeze(0).cpu().numpy()  # L
    # normalize
    if saliency.sum() > 0:
        saliency = saliency / (saliency.max() + 1e-12)
    return saliency

# ---------- main attack loop (greedy) ----------
def attack_sample_greedy(model, device, tok, api_cands, sid, label, trace, args):
    """
    Given one trace (string) for a malware sample (label==1), perform greedy saliency attack:
     - compute token ids for full trace (max_len)
     - iterate up to iter_steps:
         * compute saliency, pick topk positions not already changed
         * for each position try `cand_sample` candidates (random sample from api_cands)
         * pick the candidate that produces minimal malware prob
         * commit change and continue
     - stop when total replacements reached or malware prob < target_prob (optional)
    Returns: attacked_trace_string, n_changes, history
    """
    max_len = args.max_len
    n_replace_total = args.n_replace_total
    topk_salient = args.topk_salient
    cand_sample = args.cand_sample
    iter_steps = args.iter_steps
    target_prob = args.target_prob

    ids = tok.encode(trace, max_len=max_len)
    orig_ids = ids[:]  # list
    changed_positions = set()
    history = []
    # evaluate original prob
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        orig_prob = F.softmax(logits, dim=-1)[:,1].item()
    curr_prob = orig_prob

    for it in range(iter_steps):
        if len(changed_positions) >= n_replace_total:
            break
        # compute saliency
        sal = compute_token_saliency_for_sequence(model, device, ids, target_class=1)  # array len L
        # mask positions already changed
        cand_positions = [i for i in np.argsort(-sal)[:topk_salient] if i not in changed_positions]
        if len(cand_positions) == 0:
            break
        # try replacements for these positions (greedy order)
        for pos in cand_positions:
            if len(changed_positions) >= n_replace_total:
                break
            # sample candidate tokens to try (avoid original token)
            tries = random.sample(api_cands, min(cand_sample, len(api_cands)))
            best_token = None
            best_prob = curr_prob
            best_id = None
            for tokstr in tries:
                # map tokstr -> id (tokenizer expects mapping)
                cand_id = tok.vocab.get(tokstr, tok.unk_id)
                # skip if same
                if cand_id == ids[pos]:
                    continue
                # create candidate ids
                cand_ids = ids[:]
                cand_ids[pos] = cand_id
                with torch.no_grad():
                    logits = model(torch.tensor([cand_ids], dtype=torch.long, device=device))
                    p = F.softmax(logits, dim=-1)[:,1].item()
                if p < best_prob:
                    best_prob = p
                    best_token = tokstr
                    best_id = cand_id
            # if we found a better replacement, commit it
            if best_token is not None and best_prob < curr_prob - 1e-6:
                ids[pos] = best_id
                changed_positions.add(pos)
                history.append({"pos": int(pos), "token": best_token, "prob_before": float(curr_prob), "prob_after": float(best_prob)})
                curr_prob = best_prob
                # early stop if reached target_prob
                if target_prob is not None and curr_prob <= target_prob:
                    break
        # end for cand_positions
        # optional break if no improvement
        if len(history) == 0 and it > 0:
            break

    # build attacked trace
    attacked_trace = ids_to_trace_from_tokens(tok, ids)
    return attacked_trace, len(changed_positions), orig_prob, curr_prob, history

# ---------- main script ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--api_candidates", required=True)
    ap.add_argument("--only_malware", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_replace_total", type=int, default=12, help="max replacements per sample")
    ap.add_argument("--topk_salient", type=int, default=20, help="consider these many top salient positions")
    ap.add_argument("--cand_sample", type=int, default=200, help="try this many replacement candidates per position")
    ap.add_argument("--iter_steps", type=int, default=3, help="number of greedy iterations")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_prob", type=float, default=None, help="stop when malware prob <= target_prob")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device!="cpu" else "cpu")
    print("Device:", device)

    tok = Tokenizer(args.vocab)  # requires vocab json with .get mapping
    # attach vocab dict on tokenizer for convenience if not present
    if not hasattr(tok, "vocab"):
        try:
            tok.vocab = json.load(open(args.vocab,"r",encoding="utf-8"))
        except Exception:
            pass

    api_cands = json.load(open(args.api_candidates,"r",encoding="utf-8"))
    print("Loaded api candidates:", len(api_cands))

    # build model
    # infer vocab_size
    try:
        vocab_json = json.load(open(args.vocab, "r", encoding="utf-8"))
        vocab_size = len(vocab_json)
    except Exception:
        vocab_size = 25000
    model = load_model_simple(args.ckpt, vocab_size, device)
    model.eval()

    # read input TSV
    rows = []
    with open(args.in_tsv, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t",2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], int(parts[1]), parts[2]
            rows.append((sid, lab, trace))

    outp = Path(args.out_tsv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"n_samples": len(rows), "n_changed": 0, "samples": []}
    t0 = time.time()
    with open(outp, "w", encoding="utf-8") as fo:
        for sid, lab, trace in rows:
            if args.only_malware and lab != 1:
                fo.write(f"{sid}\t{lab}\t{trace}\n")
                continue
            attacked_trace, n_changed, before_p, after_p, history = attack_sample_greedy(model, device, tok, api_cands, sid, lab, trace, args)
            fo.write(f"{sid}\t{lab}\t{attacked_trace}\n")
            meta["samples"].append({"sid": sid, "label": lab, "changed": n_changed, "p_before": before_p, "p_after": after_p, "history": history})
            if n_changed > 0:
                meta["n_changed"] += 1
    t1 = time.time()
    meta["time_sec"] = t1 - t0
    json.dump(meta, open(str(outp) + ".meta.json", "w"), indent=2)
    print("Wrote attacked TSV and meta:", outp, outp.with_suffix(".meta.json"))
    print("Elapsed sec:", meta["time_sec"])

if __name__ == "__main__":
    main()
