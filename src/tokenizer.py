#!/usr/bin/env python3
"""
Simple tokenizer & vocab utilities for the Nebula mini training pipeline.

Usage examples:
  from tokenizer import tokenize, build_vocab, tokens_to_ids, ids_to_tokens

  # build vocab from dataset lines (list of flat traces)
  vocab = build_vocab(lines, vocab_size=20000, min_freq=1)
  save_vocab(vocab, "checkpoints/vocab.json")

  # convert
  toks = tokenize("api:ReadFile path:C:\\Windows\\Temp\\a.tmp ||| api:CreateFileW")
  ids = tokens_to_ids(toks, vocab, max_len=256)

Usage: python tokenizer.py --build_from dataset_small_2k.tsv
"""

from collections import Counter, OrderedDict
import json
import re
from typing import List, Dict

# conservative token regex: keep any non-whitespace as token (preserves "api:ReadFile" intact)
_TOKEN_RE = re.compile(r"\S+")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def tokenize(text: str) -> List[str]:
    """
    Split text into tokens. Keeps punctuation attached to tokens (fine for traces).
    """
    if not isinstance(text, str):
        text = str(text)
    return _TOKEN_RE.findall(text)

def build_vocab(texts: List[str], vocab_size: int = 20000, min_freq: int = 1, reserved: List[str] = None) -> Dict[str,int]:
    """
    Build token->id vocab from iterable of text strings.
    Returns an OrderedDict mapping token -> id with deterministic ordering:
      [reserved tokens..., most-frequent tokens...]

    reserved: list of tokens to force at the start (defaults to [PAD, UNK])
    """
    if reserved is None:
        reserved = [PAD_TOKEN, UNK_TOKEN]

    counter = Counter()
    for t in texts:
        toks = tokenize(t)
        counter.update(toks)

    # start vocab with reserved tokens
    vocab = OrderedDict()
    for tok in reserved:
        vocab[tok] = len(vocab)

    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= vocab_size:
            break

    return vocab

def save_vocab(vocab: Dict[str,int], path: str):
    """
    Save vocab as a JSON list (index -> token) for consistency with tokens_to_ids.
    Older code expects a JSON list (not a dict) so we write a list where index==id.
    """
    # vocab could be OrderedDict token->id
    inv = [None] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        inv[idx] = tok
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    print(f"Wrote vocab (size {len(inv)}) to {path}")

def load_vocab(path: str) -> Dict[str,int]:
    """
    Load vocab saved by save_vocab (JSON list -> token->id dict).
    """
    with open(path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    vocab = {tok: idx for idx, tok in enumerate(inv)}
    return vocab

def tokens_to_ids(tokens: List[str], vocab: Dict[str,int], max_len: int = 256) -> List[int]:
    """
    Convert list of tokens to list of ids (padded/truncated to max_len).
    Unknown tokens mapped to UNK token id (if present) else 1.
    Pads with PAD token id (0).
    Deterministic length for batching.
    """
    if not tokens:
        ids = []
    else:
        unk_id = vocab.get(UNK_TOKEN, 1)
        ids = [vocab.get(t, unk_id) for t in tokens]

    # truncate or pad to max_len
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        pad_id = vocab.get(PAD_TOKEN, 0)
        ids = ids + [pad_id] * (max_len - len(ids))
    return ids

def ids_to_tokens(ids: List[int], vocab: Dict[str,int]) -> List[str]:
    """
    Convert back from ids to tokens. Expects vocab as token->id mapping; we invert it.
    """
    inv = {idx: tok for tok, idx in vocab.items()}
    return [inv.get(i, UNK_TOKEN) for i in ids]

# Small CLI helpers for local debugging (not heavy)
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_from", help="text file (one trace per line) to build vocab from")
    ap.add_argument("--vocab_out", help="output vocab path (checkpoints/vocab.json)", default="checkpoints/vocab.json")
    ap.add_argument("--vocab_size", type=int, default=25000)
    ap.add_argument("--min_freq", type=int, default=1)
    args = ap.parse_args()

    if args.build_from:
        texts = []
        with open(args.build_from, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                # accept both TSV (id \t label \t trace) and plain traces
                parts = ln.split("\t")
                if len(parts) >= 3:
                    texts.append(parts[2])
                else:
                    texts.append(ln)
        vocab = build_vocab(texts, vocab_size=args.vocab_size, min_freq=args.min_freq)
        save_vocab(vocab, args.vocab_out)
        sys.exit(0)

