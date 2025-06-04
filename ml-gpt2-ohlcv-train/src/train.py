#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
train.py – Train GPT-2 time-series model on tokenized OHLCV data.
"""
import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from tqdm import tqdm

from data.dataset import TokenWindowDataset, NextTokenCollator
from network.model import GPT2TimeSeries
from utils.misc import set_seed, save_ckpt
from utils.scheduler import CosineWarmupScheduler

# ───────── Configuration ─────────
CFG = {
    "csv_dir":       "data/tokens",
    "context_size":  100,
    "batch_size":    128,
    "num_workers":   4,
    "seed":          42,
    "lr":            3e-4,
    "weight_decay":  1e-2,
    "warmup_steps":  5,
    "T_max":         95,
    "vocab_size":    2048,
    "output_dir":    "checkpoints",
    "logs_dir":      "logs",
    "train_frac":    0.9,
}
# ──────────────────────────────────

def run_epoch(model, loader, optimizer, scaler, device, epoch, mode="train"):
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    total_loss, total_count = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{mode}]", leave=False)
    for batch in pbar:
        tokens = batch["tokens"].to(device)   # [B,100]
        labels = batch["labels"].to(device)   # [B]

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=True):
                logits = model(tokens)  # [B,2048]
                loss = torch.nn.functional.cross_entropy(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        bsz = tokens.size(0)
        total_loss  += loss.item() * bsz
        total_count += bsz
        avg_loss = total_loss / total_count
        pbar.set_postfix({f"{mode}_loss": f"{avg_loss:.6f}"})

    return total_loss / total_count

def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG["output_dir"], exist_ok=True)
    os.makedirs(CFG["logs_dir"], exist_ok=True)

    # Dataset + split
    full_ds = TokenWindowDataset(CFG["csv_dir"], context_size=CFG["context_size"])
    n_total = len(full_ds)
    n_train = int(CFG["train_frac"] * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG["seed"])
    )
    collate      = NextTokenCollator()
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True,
        collate_fn=collate, drop_last=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
        collate_fn=collate, drop_last=False
    )

    # Model, optimizer, scheduler, scaler
    model     = GPT2TimeSeries(
                    vocab_size=CFG["vocab_size"],
                    context_size=CFG["context_size"]
                ).to(device)
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=CFG["lr"],
                    weight_decay=CFG["weight_decay"]
                )
    scheduler = CosineWarmupScheduler(
                    optimizer,
                    CFG["warmup_steps"],
                    CFG["T_max"]
                )
    scaler    = GradScaler()

    # Resume if checkpoint exists
    ckpts = glob.glob(os.path.join(CFG["output_dir"], "checkpoint*.pth"))
    if ckpts:
        epochs = [
            int(os.path.basename(p).split("checkpoint")[1].split(".pth")[0])
            for p in ckpts
        ]
        last_e = max(epochs)
        ckpt   = torch.load(
                    os.path.join(CFG["output_dir"], f"checkpoint{last_e}.pth"),
                    map_location=device
                )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = last_e + 1
        print(f"[INFO] Resuming from epoch {last_e}")
    else:
        start_epoch = 1

    # Training loop
    epoch = start_epoch
    while True:
        train_loss = run_epoch(
                        model, train_loader, optimizer,
                        scaler, device, epoch, mode="train"
                     )
        val_loss   = run_epoch(
                        model, val_loader, optimizer,
                        scaler, device, epoch, mode="val"
                     )

        # Checkpoint
        ckpt_path = os.path.join(
                        CFG["output_dir"], f"checkpoint{epoch}.pth"
                    )
        save_ckpt({
            "epoch":  epoch,
            "model":  model.state_dict(),
            "optim":  optimizer.state_dict(),
            "sched":  scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }, ckpt_path)

        # Logging
        with open(os.path.join(CFG["logs_dir"], "logs.txt"), "a") as f:
            f.write(
                f"Epoch {epoch}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}\n"
            )
        print(
            f"[INFO] Epoch {epoch} → "
            f"train_loss {train_loss:.6f}, val_loss {val_loss:.6f}"
        )

        scheduler.step()
        epoch += 1

if __name__ == "__main__":
    main()
