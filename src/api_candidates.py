#!/usr/bin/env python3
"""
Scan a raw traces directory and extract frequent API tokens to build an API candidate pool.

Behavior:
 - Recursively walks `raw_dir`.
 - Opens any .txt/.log/.trace files and extracts tokens matching the regex r"(api:\\S+)".
 - Counts token frequencies and outputs the top_k tokens as a JSON list.
 - Skips very large files gracefully (streaming line by line).
 - Writes output as a JSON list (suitable for Sampler and other code).

Usage:
  python api_candidates.py --raw_dir Malware-Traces --out checkpoints/api_candidates.json --top_k 500

Args:
  --raw_dir   : directory containing unzipped gw/mw folders (required)
  --out       : output JSON path (default: checkpoints/api_candidates.json)
  --top_k     : number of top APIs to output (default: 300)
  --min_freq  : minimum frequency to include (default: 2)
  --extensions: comma-separated list of file extensions to scan (default: .txt,.log)
  --max_files : optional cap on number of files to scan (debugging)

import argparse
import json
import re
from collections import Counter
from pathlib import Path



def extract_apis_from_line(line):
    # find all api: tokens in the line
    return API_RE.findall(line)

def scan_dir_for_api_candidates(raw_dir, exts=(".txt", ".log"), max_files=None, min_freq=2):
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    counter = Counter()
    files = []
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files = sorted(files)
    if max_files:
        files = files[:max_files]
    print(f"[INFO] Scanning {len(files)} files under {raw_dir} for API tokens...")
    for i, fp in enumerate(files, 1):
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    apis = extract_apis_from_line(ln)
                    for a in apis:
                        # normalize simple variants: remove trailing punctuation
                        a_norm = a.strip()
                        counter[a_norm] += 1
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
        if i % 200 == 0:
            print(f" Scanned {i}/{len(files)} files...")

    # filter by min_freq
    items = [(tok, c) for tok, c in counter.most_common() if c >= min_freq]
    return items

def write_top_k(items, out_path, top_k=300):
    top = [tok for tok, _ in items[:top_k]]
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(top, f, indent=2)
    print(f"[INFO] Wrote {len(top)} API candidates to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory with raw traces (unzipped Malware-Traces)")
    ap.add_argument("--out", default="checkpoints/api_candidates.json")
    ap.add_argument("--top_k", type=int, default=300)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--extensions", default=".txt,.log", help="comma-separated file extensions to scan")
    ap.add_argument("--max_files", type=int, default=0, help="optional cap on files to scan (0 => all)")
    args = ap.parse_args()

    exts = tuple(ext.strip().lower() if ext.strip().startswith(".") else "." + ext.strip().lower()
                 for ext in args.extensions.split(","))
    max_files = args.max_files if args.max_files and args.max_files > 0 else None

    items = scan_dir_for_api_candidates(args.raw_dir, exts=exts, max_files=max_files, min_freq=args.min_freq)
    if not items:
        print("[WARN] No API tokens found. Check raw_dir and file extensions.")
    write_top_k(items, args.out, top_k=args.top_k)

if __name__ == "__main__":
    main()
"""

# API_RE = re.compile(r"(api:[A-Za-z0-9_:\.\\\/\-\[\]\(\)]+)")

import re,json
from collections import Counter
p_func = re.compile(r"function\|([A-Za-z0-9_]+)", re.IGNORECASE)
p_mod = re.compile(r"(?:origin|target)\|([^| ]+)", re.IGNORECASE)
cnt = Counter()
with open("data/dataset_small_2k_normalized.tsv","r",encoding="utf-8",errors="ignore") as f:
    for ln in f:
        parts = ln.rstrip("\n").split("\t",2)
        if len(parts)<3: continue
        tr=parts[2]
        for m in p_func.findall(tr):
            cnt[f"function|{m}"]+=1
        for m in p_mod.findall(tr):
            # normalize paths/modules by last path component
            mod = m.split("\\")[-1].split("/")[-1]
            cnt[f"module|{mod}"]+=1
top = [tok for tok,_ in cnt.most_common(200)]
open("checkpoints/api_candidates_norm.json","w",encoding="utf-8").write(json.dumps(top,indent=2))
print('Wrote',len(top),'candidates -> checkpoints/api_candidates_norm.json')
