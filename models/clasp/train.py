"""
train.py — CLaSP training loop per docs/REIMPLEMENTATION_SPEC.md §3.

  optimizer : AdamW — lr 1e-4 (signal encoder + projections), 1e-5 (T5)
  batch 64, up to 100 epochs, early stop on val loss (patience 10), seed 42
  checkpoints -> results/checkpoints/{last,best}.pt

PILOT FIRST (time one epoch before committing to a full run):
    python -m models.clasp.train --epochs 2 --tag pilot
FULL RUN:
    python -m models.clasp.train --tag baseline_seed42
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from dataset import make_loader
from models.clasp.model import ClaspModel, ClaspConfig

CKPT_DIR = Path("results/checkpoints")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def interleaved(loaders):
    """Yield batches from all loaders in random interleaved order."""
    iters = [iter(l) for l in loaders]
    alive = list(range(len(iters)))
    while alive:
        k = random.choice(alive)
        try:
            yield next(iters[k])
        except StopIteration:
            alive.remove(k)


@torch.no_grad()
def val_loss(model, loaders, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in (b for l in loaders for b in l):
        batch["series"] = batch["series"].to(device)
        batch["series_mask"] = batch["series_mask"].to(device)
        loss, _ = model(batch)
        bs = batch["series"].size(0)
        tot += loss.item() * bs
        n += bs
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-truce", type=int, default=64)
    ap.add_argument("--batch-sushi", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-text", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    truce = ("truce_stock", "truce_synth")
    train_loaders = [
        make_loader(splits=("train",), datasets=truce,
                    batch_size=args.batch_truce, shuffle=True),
        make_loader(splits=("train",), datasets=("sushi",),
                    batch_size=args.batch_sushi, shuffle=True),
    ]
    val_loaders = [
        make_loader(splits=("val",), datasets=truce,
                    batch_size=args.batch_truce, shuffle=False),
        make_loader(splits=("val",), datasets=("sushi",),
                    batch_size=args.batch_sushi, shuffle=False),
    ]
    n_tr = sum(len(l.dataset) for l in train_loaders)
    n_va = sum(len(l.dataset) for l in val_loaders)
    print(f"train pairs: {n_tr}, val pairs: {n_va} "
          f"(per-dataset batches: truce {args.batch_truce}, sushi {args.batch_sushi})")

    model = ClaspModel(ClaspConfig()).to(device)
    text_params = list(model.text_encoder.parameters())
    text_ids = {id(p) for p in text_params}
    other_params = [p for p in model.parameters() if id(p) not in text_ids]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": text_params, "lr": args.lr_text}],
        weight_decay=0.01)

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history, best_val, bad_epochs = [], float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in interleaved(train_loaders):
            batch["series"] = batch["series"].to(device)
            batch["series_mask"] = batch["series_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device,
                                    enabled=(device == "cuda")):
                loss, _ = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            bs = batch["series"].size(0)
            tot += loss.item() * bs
            n += bs
        tr = tot / n
        vl = val_loss(model, val_loaders, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr,
                        "val_loss": vl, "seconds": round(dt, 1)})
        print(f"epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}  ({dt:.0f}s)")

        state = {"model": model.state_dict(), "epoch": epoch,
                 "val_loss": vl, "args": vars(args)}
        torch.save(state, CKPT_DIR / f"last_{args.tag}.pt")
        if vl < best_val - 1e-4:
            best_val, bad_epochs = vl, 0
            torch.save(state, CKPT_DIR / f"best_{args.tag}.pt")
            print(f"  new best (val {vl:.4f}) -> best_{args.tag}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} "
                      f"(no val improvement for {args.patience})")
                break

    with open(CKPT_DIR / f"history_{args.tag}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best val loss {best_val:.4f}. "
          f"history -> history_{args.tag}.json")


if __name__ == "__main__":
    main()
