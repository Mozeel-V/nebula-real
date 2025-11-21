import argparse, json, random
from pathlib import Path
from collections import defaultdict

def load_api_candidates(path):
    return json.load(open(path, "r", encoding="utf-8"))

def read_sampler_input(sampler_tsv):
    """
    Expect TSV: sid \t label \t trace
    """
    rows = []
    with open(sampler_tsv, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t",2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], int(parts[1]), parts[2]
            rows.append((sid, lab, trace))
    return rows

def replace_tokens_in_trace(trace, replacements, n_replace):
    toks = trace.split()
    L = len(toks)
    if L == 0:
        return trace
    # choose n_replace random positions (avoid first and last positions heuristically)
    positions = list(range(0, L))
    if len(positions) <= n_replace:
        chosen = positions
    else:
        chosen = random.sample(positions, n_replace)
    for pos in chosen:
        toks[pos] = random.choice(replacements)
    return " ".join(toks)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampler_in", required=True, help="TSV of samples to attack (sid\\tlabel\\ttrace)")
    p.add_argument("--api_candidates", required=True, help="json list of replacement tokens (in-vocab)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_replace", type=int, default=12)
    p.add_argument("--only_malware", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    reps = load_api_candidates(args.api_candidates)
    rows = read_sampler_input(args.sampler_in)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for sid, lab, trace in rows:
            if args.only_malware and lab != 1:
                # write original
                f.write(f"{sid}\t{lab}\t{trace}\n")
                continue
            adv_trace = replace_tokens_in_trace(trace, reps, args.n_replace)
            f.write(f"{sid}\t{lab}\t{adv_trace}\n")
    print("Wrote attacked dataset to", args.out)

if __name__ == "__main__":
    main()
