#!/usr/bin/env python3
"""
Saliency-guided replacement attack with an integrated Sampler tuned for
function|... and module|... tokens.

Outputs:
 - adversarial TSV (same format sid \t label \t trace)
 - small JSON report with stats

This script will:
 - load saliency positions (sid -> [token_idx,...])
 - load api candidates (here function|NAME and/or module|NAME tokens)
 - instantiate a Sampler that mixes effectiveness and uniform exploration
 - for each malware sample, pick up to n_replace top saliency indices and replace
   them with sampled events (function:Name arg:NUM or module:Name)
"""
import argparse, json, random, time
from pathlib import Path
from typing import List, Optional
from collections import Counter

# ------------------------
# Simple Sampler (integrated)
# ------------------------
import math
import numpy as np

class SimpleSampler:
    """
    A compact sampler that builds a probability distribution over tokens (function|X, module|Y)
    from:
      - api_candidates: list[str]
      - api_effectiveness: optional list of {"api": token, "mean_delta": float}
      - benign_counts: optional map token->count (not used here, but kept)
    Combines signals with alpha/beta/gamma weights and temperature scaling.
    """
    def __init__(self,
                 api_candidates_path: str,
                 api_effectiveness_path: Optional[str] = None,
                 alpha: float = 0.7, beta: float = 0.25, gamma: float = 0.05,
                 temp: float = 0.9, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.apis = json.load(open(api_candidates_path, "r", encoding="utf-8"))
        # load effectiveness list -> map
        eff_list = []
        if api_effectiveness_path and Path(api_effectiveness_path).exists():
            eff_list = json.load(open(api_effectiveness_path, "r", encoding="utf-8"))
        eff_map = {}
        for item in eff_list:
            if isinstance(item, dict):
                key = item.get("api")
                if key:
                    eff_map[key] = float(item.get("mean_delta", 0.0))
        eff_vec = np.array([eff_map.get(a, 0.0) for a in self.apis], dtype=float)
        # freq placeholder (not used)
        freq_vec = np.zeros_like(eff_vec)
        # normalize
        def normalize(v):
            if v.size == 0:
                return v
            mn, mx = float(v.min()), float(v.max())
            if abs(mx - mn) < 1e-12:
                return np.zeros_like(v)
            return (v - mn) / (mx - mn + 1e-12)
        eff_n = normalize(eff_vec)
        freq_n = normalize(freq_vec)
        weights = alpha * eff_n + beta * freq_n + gamma
        weights = np.clip(weights, 0.0, None)
        if temp != 1.0:
            safe = np.clip(weights, 1e-12, None)
            logits = np.log(safe)
            scaled = np.exp(logits / float(temp))
            weights = scaled
        if weights.sum() <= 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / float(weights.sum())
        # mix small uniform epsilon
        eps = 0.01
        uniform = np.ones_like(probs) / len(probs)
        probs = (1 - eps) * probs + eps * uniform
        self.probs = probs.tolist()

    def sample_token(self) -> str:
        return random.choices(self.apis, weights=self.probs, k=1)[0]

    def sample_event_from_token(self, token: str, idx: int = 0) -> str:
        """
        Build event strings appropriate for normalized traces:
          - function|Name -> function:Name arg:NUM
          - module|Name   -> module:Name
        """
        if token.startswith("function|"):
            fname = token.split("|", 1)[1]
            return f"function:{fname} arg:NUM"
        if token.startswith("module|"):
            mod = token.split("|", 1)[1]
            return f"module:{mod}"
        # fallback: preserve token as is
        return token

    def export_probs(self, path: str):
        data = [{"api": a, "prob": float(p)} for a, p in zip(self.apis, self.probs)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

# ------------------------
# Attack logic
# ------------------------
def load_tsv(path: str):
    samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln: continue
            parts = ln.split("\t", 2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], int(parts[1]), parts[2]
            samples.append((sid, lab, trace))
    return samples

def replace_token_at_index(flat_trace: str, token_idx: int, new_event: str) -> str:
    """
    Replace the token at index `token_idx` in the normalized trace.
    We treat event separator " ||| " as the high-level event delimiter.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="normalized TSV (sid\\tlabel\\ttrace)")
    ap.add_argument("--saliency", required=True, help="saliency JSON (sid -> [token_idx,...])")
    ap.add_argument("--api_pool", required=True, help="api candidates JSON (function|... or module|...)")
    ap.add_argument("--api_effect", default=None, help="optional api effectiveness JSON")
    ap.add_argument("--out", default="results/attacks/saliency_sampler_adversarial_norm.tsv")
    ap.add_argument("--report", default="results/attacks/saliency_attack_report_norm.json")
    ap.add_argument("--n_replace", type=int, default=6, help="max replacements per malware sample")
    ap.add_argument("--k", type=int, default=32, help="top-k saliency positions to consider")
    ap.add_argument("--sample_limit", type=int, default=0, help="limit number of malware samples to attack (0 => all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=0.9)
    args = ap.parse_args()

    # deterministic-ish
    random.seed(args.seed)

    samples = load_tsv(args.dataset)
    salmap = json.load(open(args.saliency, "r", encoding="utf-8"))
    api_candidates = json.load(open(args.api_pool, "r", encoding="utf-8"))

    sampler = SimpleSampler(api_candidates_path=args.api_pool,
                            api_effectiveness_path=args.api_effect,
                            alpha=args.alpha, beta=args.beta, gamma=args.gamma, temp=args.temp, seed=args.seed)

    # prepare output
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fout = open(outp, "w", encoding="utf-8")

    stats = {"total_samples": len(samples), "total_malware": 0, "attacked": 0, "skipped": 0, "n_replace": args.n_replace, "k": args.k}
    used = 0
    limit = args.sample_limit if args.sample_limit > 0 else None

    for sid, lab, trace in samples:
        if lab != 1:
            fout.write(f"{sid}\t{lab}\t{trace}\n")
            continue
        stats["total_malware"] += 1
        if limit is not None and used >= limit:
            fout.write(f"{sid}\t{lab}\t{trace}\n")
            continue
        used += 1
        # get saliency positions
        pos_list = salmap.get(str(sid), salmap.get(int(sid), []))
        if not pos_list:
            # fallback: pick first k event indices
            parts = trace.split(" ||| ") if " ||| " in trace else trace.split()
            pos_list = list(range(min(len(parts), args.k)))
        pos_list = pos_list[: args.k]
        # choose positions to replace
        nrep = min(len(pos_list), args.n_replace)
        chosen = random.sample(pos_list, nrep) if nrep > 0 else []
        adv = trace
        replacements = []
        for i, pos in enumerate(chosen):
            token = sampler.sample_token()
            ev = sampler.sample_event_from_token(token, idx=i)
            adv = replace_token_at_index(adv, pos, ev)
            replacements.append({"pos": int(pos), "token": token, "event": ev})
        fout.write(f"{sid}\t{lab}\t{adv}\n")
        stats["attacked"] += 1
        if stats["attacked"] % 50 == 0:
            print(f"[INFO] attacked {stats['attacked']} malware samples (used {used})")

    fout.close()
    stats["elapsed_s"] = time.time() - stats.get("start_time", time.time())
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as rf:
        json.dump(stats, rf, indent=2)
    print(f"Wrote adversarial TSV -> {outp}")
    print(f"Wrote report -> {args.report}")

if __name__ == "__main__":
    import argparse
    main()
