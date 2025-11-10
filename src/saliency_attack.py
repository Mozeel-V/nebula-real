#!/usr/bin/env python3
"""
Saliency-guided replacement attack (patched to use SaliencyWeighted Sampler).

Saves an adversarial TSV and a small JSON report with stats.

Behavior:
 - For each malware sample (label==1) select up to `n_replace` token/event positions
   from the saliency list (or fallback top-k positions).
 - For each chosen position, propose a replacement using the SaliencyWeighted Sampler
   if available, otherwise fall back to uniform api sampling.
 - Replacement strings are realistic event tokens (sampler.sample_api_event()).
 - Writes adversarial TSV and a JSON report summarizing how many samples were attacked.

Usage example:
  python saliency_attack.py \
    --dataset dataset_small_2k.tsv \
    --saliency results/saliency_positions.json \
    --api_pool checkpoints/api_candidates.json \
    --api_effect checkpoints/api_effectiveness.json \
    --out results/attacks/saliency_sampler_adversarial.tsv \
    --n_replace 8 --k 32 --use_sampler
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional

from saliency_weighted_sampler import Sampler  # type: ignore

# Fallback simple uniform API event generator
def sample_event_from_api(api_token: str, idx: int = 0) -> str:
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

def load_tsv(path: str):
    samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t", 2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], int(parts[1]), parts[2]
            samples.append((sid, lab, trace))
    return samples

def write_report(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="input TSV (sid\\tlabel\\ttrace)")
    ap.add_argument("--saliency", required=True, help="saliency JSON (sid -> [idx,...])")
    ap.add_argument("--api_pool", required=True, help="api candidates JSON list")
    ap.add_argument("--api_effect", default=None, help="api effectiveness JSON (optional, used by sampler)")
    ap.add_argument("--out", default="results/attacks/saliency_sampler_adversarial.tsv")
    ap.add_argument("--report", default="results/attacks/saliency_attack_report.json")
    ap.add_argument("--n_replace", type=int, default=6, help="number of replacements per malware sample")
    ap.add_argument("--k", type=int, default=32, help="top-k saliency positions to consider")
    ap.add_argument("--sample_limit", type=int, default=500, help="limit number of malware samples to attack (0 => all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_sampler", action="store_true", help="use SaliencyWeighted Sampler if available")
    ap.add_argument("--sampler_alpha", type=float, default=0.7)
    ap.add_argument("--sampler_beta", type=float, default=0.25)
    ap.add_argument("--sampler_gamma", type=float, default=0.05)
    ap.add_argument("--sampler_temp", type=float, default=0.9)
    args = ap.parse_args()

    random.seed(args.seed)

    samples = load_tsv(args.dataset)
    salmap = json.load(open(args.saliency, "r", encoding="utf-8"))
    api_pool = json.load(open(args.api_pool, "r", encoding="utf-8"))

    # instantiate sampler if requested and available
    sampler = None
    if args.use_sampler:
        try:
            # try constructing sampler using paths
            sampler = Sampler(
                api_candidates=args.api_pool,
                api_effectiveness=args.api_effect,
                benign_counts=None,
                alpha=args.sampler_alpha,
                beta=args.sampler_beta,
                gamma=args.sampler_gamma,
                temp=args.sampler_temp,
                seed=args.seed
            )
            print("[INFO] Using SaliencyWeighted Sampler")
            # optional: export sampler probs for debugging
            try:
                sampler.export_probs("results/attacks/sampler_probs.json")
            except Exception:
                pass
        except Exception as e:
            print("[WARN] Failed to instantiate Sampler, falling back to uniform sampling:", e)
            sampler = None
    else:
        print("[INFO] Using uniform API sampling.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    attacked = 0
    total_malware = sum(1 for _, lab, _ in samples if lab == 1)
    limit = args.sample_limit if args.sample_limit > 0 else total_malware
    used = 0
    stats = {
        "total_samples": len(samples),
        "total_malware": total_malware,
        "attacked": 0,
        "skipped": 0,
        "start_time": time.time(),
        "seed": args.seed,
        "n_replace": args.n_replace,
        "k": args.k,
        "use_sampler": bool(sampler is not None)
    }

    for sid, lab, trace in samples:
        if lab != 1:
            out_f.write(f"{sid}\t{lab}\t{trace}\n")
            continue

        if used >= limit:
            out_f.write(f"{sid}\t{lab}\t{trace}\n")
            continue

        used += 1
        # get saliency indices for this sample
        pos_list = salmap.get(str(sid), salmap.get(int(sid), []))
        if not pos_list:
            # fallback: event/token positions
            parts = trace.split(" ||| ") if " ||| " in trace else trace.split()
            pos_list = list(range(min(len(parts), args.k)))
        pos_list = pos_list[: args.k]

        nrep = min(len(pos_list), args.n_replace)
        chosen = random.sample(pos_list, nrep) if nrep > 0 else []
        adv = trace
        replaced_positions = []
        replacements = []
        for i, pos in enumerate(chosen):
            if sampler:
                try:
                    ev = sampler.sample_api_event(idx=i)
                except Exception:
                    api = random.choice(api_pool)
                    ev = sample_event_from_api(api, idx=i)
            else:
                api = random.choice(api_pool)
                ev = sample_event_from_api(api, idx=i)
            adv = replace_token_at_index(adv, pos, ev)
            replaced_positions.append(int(pos))
            replacements.append(ev)

        out_f.write(f"{sid}\t{lab}\t{adv}\n")
        attacked += 1
        stats["attacked"] += 1
        # light logging per 50
        if attacked % 50 == 0:
            print(f"[INFO] attacked {attacked}/{limit} malware samples")

    out_f.close()
    stats["end_time"] = time.time()
    stats["elapsed_s"] = stats["end_time"] - stats["start_time"]
    print(f"Wrote {out_path} (attacked {attacked} malware samples). elapsed {stats['elapsed_s']:.1f}s")
    write_report(args.report, stats)
    print("Wrote report ->", args.report)


if __name__ == "__main__":
    main()
