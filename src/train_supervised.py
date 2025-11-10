#!/usr/bin/env python3
"""
Train a small Nebula-like classifier on TSV dataset (id \t label \t trace).

Saves:
 - checkpoints/{epoch:%03d}.pt  (state_dict + config + epoch + optimizer state)
 - checkpoints/best.pt          (best validation model)
 - checkpoints/vocab.json       (if built here)
 - logs/train_logs.json         (per-epoch metrics)

Usage: python train_supervised.py --data_file dataset_small_2k.tsv
"""

import argparse
import json
import math
import random
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

from tokenizer import tokenize, build_vocab, save_vocab, load_vocab, tokens_to_ids, PAD_TOKEN, UNK_TOKEN
from nebula_model import NebulaTiny


# -----------------------
# Dataset + Collate
# -----------------------
class TraceDataset(Dataset):
    def __init__(self, tsv_path):
        self.samples = []
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                parts = ln.split("\t", 2)
                if len(parts) < 3:
                    continue
                sid, lab, trace = parts[0], int(parts[1]), parts[2]
                self.samples.append((sid, lab, trace))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, lab, trace = self.samples[idx]
        return sid, lab, trace

def collate_fn(batch, vocab, max_len):
    # batch: list of (sid, lab, trace)
    sids = [b[0] for b in batch]
    labs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    token_lists = [tokenize(b[2]) for b in batch]
    ids = [tokens_to_ids(toks, vocab, max_len=max_len) for toks in token_lists]
    x = torch.tensor(ids, dtype=torch.long)
    return sids, x, labs

# -----------------------
# Training loop
# -----------------------
def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    loss_f = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for _, x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_f(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return running_loss / total if total else 0.0, correct / total if total else 0.0

def train_epoch(model, opt, loader, device):
    model.train()
    loss_f = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    correct = 0
    for _, x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_f(logits, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return running_loss / total if total else 0.0, correct / total if total else 0.0

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True, help="TSV dataset (id \\t label \\t trace)")
    p.add_argument("--vocab", default="checkpoints/vocab.json", help="path to vocab json (list)")
    p.add_argument("--build_vocab", action="store_true", help="build vocab from dataset_small if vocab not present")
    p.add_argument("--vocab_size", type=int, default=25000)
    p.add_argument("--min_freq", type=int, default=1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--save_every", type=int, default=1, help="save checkpoint each N epochs")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # build or load vocab
    if args.build_vocab or not Path(args.vocab).exists():
        print("Building vocab from dataset...")
        # read traces quickly
        traces = []
        with open(args.data_file, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t", 2)
                if len(parts) >= 3:
                    traces.append(parts[2])
        vocab_map = build_vocab(traces, vocab_size=args.vocab_size, min_freq=args.min_freq)
        save_vocab(vocab_map, args.vocab)
    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"Loaded vocab size: {vocab_size}")

    # dataset and split
    ds = TraceDataset(args.data_file)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"Dataset size: {len(ds)} train={len(train_ds)} val={len(val_ds)}")

    # dataloaders
    collate = lambda batch: collate_fn(batch, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # model config (choose small model for CPU)
    cfg = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "ff": 256,
        "max_len": args.max_len,
        "num_classes": 2,
        "chunk_size": 0
    }

    model = NebulaTiny(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["ff"],
        max_len=cfg["max_len"],
        num_classes=cfg["num_classes"],
        chunk_size=cfg["chunk_size"]
    )

    model.to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    logs = {"epochs": []}

    total_start = time()
    for epoch in range(1, args.epochs + 1):
        t0 = time()
        train_loss, train_acc = train_epoch(model, opt, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        epoch_time = time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")
        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_s": epoch_time
        })

        # save checkpoint every save_every
        if epoch % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
                "config": cfg,
                "vocab_size": vocab_size
            }
            ckpt_path = outdir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = outdir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim_state": opt.state_dict(),
                "config": cfg
            }, best_path)
            print("Saved best model:", best_path)

        # write logs to disk each epoch
        with open(outdir / "train_logs.json", "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

    total_time = time() - total_start
    print(f"Training finished in {total_time:.1f}s. Best val acc: {best_val_acc:.4f}")
    # final save
    final_path = outdir / "final.pt"
    torch.save({"epoch": args.epochs, "model": model.state_dict(), "config": cfg}, final_path)
    print("Saved final model:", final_path)


if __name__ == "__main__":
    main()
