import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tokenizer import Tokenizer
from nebula_model import NebulaTiny, NebulaCLS

# -----------------------
# Simple Dataset
# -----------------------
class TraceDataset(Dataset):
    def __init__(self, tsv_path, vocab_path, max_len=256, normalize=False):
        # expects lines: sid \t label \t trace
        self.rows = []
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                parts = ln.split("\t", 2)
                if len(parts) < 3:
                    continue
                sid, lab, trace = parts[0], int(parts[1]), parts[2]
                self.rows.append((sid, lab, trace))
        # tokenizer expects a vocab json path
        self.tok = Tokenizer(vocab_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sid, lab, trace = self.rows[idx]
        ids = self.tok.encode(trace, max_len=self.max_len)  # expect list[int]
        return {"sid": sid, "label": lab, "input_ids": torch.tensor(ids, dtype=torch.long)}

def collate_batch(batch):
    # pad to max in batch with 0 (assumes padding_idx=0)
    maxlen = max([len(item["input_ids"]) for item in batch])
    ids = []
    labels = []
    sids = []
    for item in batch:
        seq = item["input_ids"]
        pad_len = maxlen - seq.shape[0]
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        ids.append(seq.unsqueeze(0))
        labels.append(item["label"])
        sids.append(item["sid"])
    ids = torch.cat(ids, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"sids": sids, "input_ids": ids, "labels": labels}

def eval_epoch(model, loader, device, criterion):

    if loader is None:
        return None, None, None, None, None

    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

            probs = torch.softmax(logits, dim=-1)[:, 1]  # malware prob
            preds = (probs >= 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if total == 0:
        return None, None, None, None, None

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    val_loss = total_loss / total
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return val_loss, acc, prec, rec, f1

# -----------------------
# Training loop
# -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # dataset + loader
    ds = TraceDataset(args.data, args.vocab, max_len=args.max_len)
    # simple shuffle and split if val_frac provided
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)
    nval = int(n * args.val_frac) if args.val_frac > 0 else 0
    if nval > 0:
        val_idx = idxs[:nval]
        train_idx = idxs[nval:]
    else:
        val_idx = []
        train_idx = idxs

    print(f"[DEBUG] total samples={n}, train={len(train_idx)}, val={len(val_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if nval > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch) if val_ds else None

    # model
    vocab_size = len(json.load(open(args.vocab, "r", encoding="utf-8")))
    print(f"[DEBUG] vocab_size={vocab_size}")
    if args.model_type == "nebula_cls":
        model = NebulaCLS(vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                          dim_feedforward=args.ff, max_len=args.max_len, num_classes=2)
    else:
        model = NebulaTiny(vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                           dim_feedforward=args.ff, max_len=args.max_len, num_classes=2)

    model.to(device)

    # class weights: if pos_weight provided, interpret as weight for class 0 (goodware) for convenience
    if args.weight0 is None and args.pos_weight is not None:
        weight0 = float(args.pos_weight)
        weight1 = 1.0
    else:
        weight0 = float(args.weight0) if args.weight0 is not None else 1.0
        weight1 = float(args.weight1) if args.weight1 is not None else 1.0

    class_weight = torch.tensor([weight0, weight1], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    osaved = Path(args.out_dir)
    osaved.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        first_batch = True
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if first_batch:
                print(f"[DEBUG][Epoch {epoch}] batch ids shape={ids.shape}, labels shape={labels.shape}")
                first_batch = False
            logits = model(ids)  # [B,2]
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = labels.shape[0]
            total_loss += float(loss.item()) * bs
            total += bs

        avg_loss = total_loss / (total + 1e-12)
        if val_loader is not None:
            val_loss, val_acc, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, device, criterion)
            print(f"[Epoch {epoch}] train_loss={avg_loss:.6f} "
                  f"| val_loss={val_loss:.6f} acc={val_acc:.3f} "
                  f"prec={val_prec:.3f} rec={val_rec:.3f} f1={val_f1:.3f}")
        else:
            val_loss = None
            print(f"[Epoch {epoch}] train_loss={avg_loss:.6f} (no validation set)")

        # validation
        '''
        val_loss = None
        if val_loader:
            model.eval()
            vtot = 0.0; vcount = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(ids)
                    loss = criterion(logits, labels)
                    vtot += float(loss.item()) * labels.shape[0]
                    vcount += labels.shape[0]
            val_loss = vtot / (vcount + 1e-12)
            print(f"   val_loss={val_loss:.6f}")
        '''

        # save checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": {
                "vocab": args.vocab,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "ff": args.ff,
                "max_len": args.max_len,
                "class_weight": [float(weight0), float(weight1)]
            }
        }
        torch.save(ckpt, osaved / f"checkpoint_epoch{epoch}.pt")
        # keep best
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, osaved / "best.pt")

    print("Training finished. Best val loss:", best_val_loss)
    # save final
    torch.save({"epoch": args.epochs, "model": model.state_dict(), "config": ckpt["config"]}, osaved / "final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="TSV: sid\\tlabel\\ttrace")
    parser.add_argument("--vocab", required=True, help="vocab json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.1)
    # class weight args:
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="legacy: if set, interpreted as weight for class0 (benign). Prefer weight0/weight1.")
    parser.add_argument("--weight0", type=float, default=None, help="weight for class 0 (goodware).")
    parser.add_argument("--weight1", type=float, default=None, help="weight for class 1 (malware).")
    parser.add_argument("--model_type", choices=["nebula_tiny", "nebula_cls"], default="nebula_tiny")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU even if CUDA available")
    args = parser.parse_args()

    # create out dir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train(args)
