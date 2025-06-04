#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
eval_conf.py – Evaluate confidence vs. accuracy & direction as a survival curve.

Runs your GPT-2 time-series model on the test set, collects each sample's
top-1 prediction confidence, correctness, and direction correctness, then
for each 5%-step confidence threshold (0.00, 0.05, …, 0.95) computes:
  • coverage   – fraction of all examples with confidence ≥ threshold
  • confidence – the threshold itself (lower bound of the percentile)
  • accuracy   – fraction of those ≥ threshold whose top-1 is correct
  • direction  – fraction of those ≥ threshold whose up/down is correct

Saves results to `logs/test/confidence.csv`.
"""
import os
import json
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import TokenWindowDataset, NextTokenCollator
from network.model import GPT2TimeSeries

# ── CONFIG ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join("checkpoints", "checkpoint3.pth")
VOCAB_PATH      = os.path.join("models",    "vocab.json")
TEST_CSV_DIR    = os.path.join("data",      "test", "tokens")
CONTEXT_SIZE    = 100
BATCH_SIZE      = 128
NUM_WORKERS     = 4
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGS_DIR        = os.path.join("logs", "test")
CSV_PATH        = os.path.join(LOGS_DIR, "confidence.csv")
BIN_WIDTH       = 0.05
NUM_BINS        = int(1.0 / BIN_WIDTH)  # 20 thresholds: 0.00, 0.05, …, 0.95
# ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Load vocab.json for direction decoding
    with open(VOCAB_PATH, "r") as f:
        vocab_map = json.load(f)

    # Load checkpoint & build model
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    vocab_size = state["lm_head.weight"].shape[0]
    model = GPT2TimeSeries(vocab_size=vocab_size, context_size=CONTEXT_SIZE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # Prepare test DataLoader
    test_ds = TokenWindowDataset(TEST_CSV_DIR, context_size=CONTEXT_SIZE)
    collate = NextTokenCollator()
    loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate
    )

    # Collect per-sample metrics
    confidences = []
    corrects    = []
    dir_correct = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(DEVICE)  # [B, L]
            labels = batch["labels"].to(DEVICE)  # [B]
            logits = model(tokens)               # [B, V]
            probs, preds1 = F.softmax(logits, dim=1).max(dim=1)  # [B], [B]

            for prob, pred, true in zip(probs, preds1, labels):
                confidences.append(prob.item())
                corrects.append(int(pred.item() == true.item()))
                # direction
                o_t, c_t = vocab_map[str(true.item())][0], vocab_map[str(true.item())][3]
                o_p, c_p = vocab_map[str(pred.item())][0], vocab_map[str(pred.item())][3]
                dir_correct.append(int((c_p >= o_p) == (c_t >= o_t)))

    total = len(confidences)

    # Write out CSV
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["coverage", "confidence", "accuracy", "direction"])
        for i in range(NUM_BINS):
            threshold = i * BIN_WIDTH
            # select all coverage with conf ≥ threshold
            idxs = [j for j, c in enumerate(confidences) if c >= threshold]
            cnt = len(idxs)
            frac = cnt / total if total else 0.0
            # compute accuracy & direction on survivors
            acc = sum(corrects[j] for j in idxs) / cnt if cnt else 0.0
            dir_acc = sum(dir_correct[j] for j in idxs) / cnt if cnt else 0.0

            writer.writerow([
                f"{frac:.6f}",       # fraction (0–1) of coverage ≥ threshold
                f"{threshold:.2f}",  # threshold lower bound
                f"{acc:.6f}",        # accuracy among those coverage
                f"{dir_acc:.6f}"     # direction accuracy among those coverage
            ])

    print(f"Wrote confidence survival curve → {CSV_PATH}")

if __name__ == "__main__":
    main()
