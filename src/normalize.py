#!/usr/bin/env python3
"""
Normalize traces by replacing high-variance fields:
 - timestamps -> TIMESTAMP
 - pid/tid -> PID/TID
 - file paths -> PATH
 - hex numbers and large integers -> NUM
 - GUIDs/UUID-like -> GUID
 - IPs -> IP
 - version numbers -> VER

This reduces unique tokens and makes vocab meaningful.
Input TSV (sid \t label \t trace). Output same format with normalized trace.
"""
import re
import argparse

# regex patterns
TS_RE = re.compile(r"\b\d{9,}\b")  # large ints (timestamps)
PID_RE = re.compile(r"\bpid\|?\:?(\d+)\b", flags=re.IGNORECASE)
TID_RE = re.compile(r"\btid\|?\:?(\d+)\b", flags=re.IGNORECASE)
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PATH_RE = re.compile(r"[A-Za-z]:(?:\\\\|/)[^\s\|]+")  # simple Windows path
NUM_RE = re.compile(r"\b\d+\b")
GUID_RE = re.compile(r"\b[0-9a-fA-F]{8}\-[0-9a-fA-F\-]{8,}\b")
SEP = " ||| "

def normalize_trace(trace: str) -> str:
    # replace timestamps (very large numbers) first
    t = TS_RE.sub("TIMESTAMP", trace)
    # replace IPs
    t = IP_RE.sub("IP", t)
    # replace hex
    t = HEX_RE.sub("HEX", t)
    # replace GUIDs
    t = GUID_RE.sub("GUID", t)
    # replace PIDs/TIDs (pattern like pid|123 or pid:123 or pid=123)
    t = PID_RE.sub("PID", t)
    t = TID_RE.sub("TID", t)
    # normalize paths
    t = PATH_RE.sub("PATH", t)
    # reduce long numbers (but keep small benign numbers e.g., feature IDs)
    # we replace numbers longer than 3 digits first via TS_RE handled; now any number -> NUM
    t = NUM_RE.sub("NUM", t)
    # collapse repeated whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # ensure event separator spacing consistent
    t = t.replace("|||", SEP.strip())
    # if original used ' ||| ' we will rejoin by SEP later in windowing
    return t

def process_file(input_tsv, output_tsv):
    n_in = n_out = 0
    with open(input_tsv, "r", encoding="utf-8", errors="ignore") as fi, \
         open(output_tsv, "w", encoding="utf-8") as fo:
        for ln in fi:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t", 2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], parts[1], parts[2]
            norm = normalize_trace(trace)
            fo.write(f"{sid}\t{lab}\t{norm}\n")
            n_in += 1
    print(f"Wrote {n_in} normalized lines to {output_tsv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", default="data/dataset_small_2k_normalized.tsv")
    args = ap.parse_args()
    process_file(args.infile, args.outfile)
