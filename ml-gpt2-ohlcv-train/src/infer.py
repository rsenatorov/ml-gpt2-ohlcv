#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
infer.py – Evaluate GPT-2 time-series model on test data, including
directional (up/down) accuracy and 4-candle trend accuracy.

Computes:
  • accuracy@1, @5, @10
  • balanced accuracy
  • macro/micro precision, recall, F1
  • total_up, total_down, correct_up, correct_down, direction_accuracy
  • total_windows_4step, correct_trend4, trend4_accuracy

Saves all metrics to JSON and text summary in logs/test/.
"""
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import TokenWindowDataset, NextTokenCollator
from network.model import GPT2TimeSeries
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score

# ── CONFIG ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join("checkpoints_kelly", "kelly_epoch1.pth")
VOCAB_PATH      = os.path.join("models", "vocab.json")
TEST_CSV_DIR    = os.path.join("data", "test", "tokens")
CONTEXT_SIZE    = 100
BATCH_SIZE      = 128
NUM_WORKERS     = 4
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGS_DIR        = os.path.join("logs", "test")
TOP_K           = [1, 5, 10]
# ───────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_k = {k: 0 for k in TOP_K}
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)   # [B, L]
            labels = batch["labels"].to(device)   # [B]
            logits = model(tokens)                # [B, vocab_size]

            # top-1
            preds1 = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds1.cpu().tolist())
            total += labels.size(0)

            # top-K
            maxk = max(TOP_K)
            topk = logits.topk(maxk, dim=1).indices  # [B, maxk]
            for k in TOP_K:
                correct_k[k] += (
                    (topk[:, :k] == labels.unsqueeze(1))
                    .any(dim=1).sum().item()
                )

    metrics = {"total_samples": total}
    for k in TOP_K:
        metrics[f"accuracy@{k}"] = correct_k[k] / total if total else 0.0

    # classification metrics (top-1)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    metrics.update({
        "precision_macro":    p_mac,
        "recall_macro":       r_mac,
        "f1_macro":           f1_mac,
        "precision_micro":    p_mic,
        "recall_micro":       r_mic,
        "f1_micro":           f1_mic,
        "balanced_accuracy":  bal_acc,
    })

    return metrics, y_true, y_pred

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1) Load test DataLoader
    test_ds = TokenWindowDataset(TEST_CSV_DIR, context_size=CONTEXT_SIZE)
    collate = NextTokenCollator()
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate
    )

    # 2) Load checkpoint → infer vocab_size
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    vocab_size = state["lm_head.weight"].shape[0]  # [vocab_size, d_model]

    # 3) Build model & load weights
    model = GPT2TimeSeries(vocab_size=vocab_size, context_size=CONTEXT_SIZE).to(DEVICE)
    model.load_state_dict(state)

    # 4) Run immediate inference
    print(f"Running inference on {len(test_ds)} samples…")
    metrics, y_true, y_pred = evaluate(model, test_loader, DEVICE)

    # 5) Load vocab.json to decode tokens → [O,H,L,C,V]
    with open(VOCAB_PATH, "r") as f:
        vocab_map = json.load(f)

    # 6) Compute single-step directional stats (argmax only)
    true_dirs = []
    pred_dirs = []
    for t, p in zip(y_true, y_pred):
        o_true, c_true = vocab_map[str(t)][0], vocab_map[str(t)][3]
        o_pred, c_pred = vocab_map[str(p)][0], vocab_map[str(p)][3]
        true_dirs.append(int(c_true >= o_true))
        pred_dirs.append(int(c_pred >= o_pred))

    total_up   = sum(true_dirs)
    total_down = len(true_dirs) - total_up
    correct_up   = sum(1 for td, pd in zip(true_dirs, pred_dirs) if td == 1 and pd == 1)
    correct_down = sum(1 for td, pd in zip(true_dirs, pred_dirs) if td == 0 and pd == 0)
    dir_acc      = (correct_up + correct_down) / len(true_dirs) if true_dirs else 0.0

    metrics.update({
        "total_up":           total_up,
        "total_down":         total_down,
        "correct_up":         correct_up,
        "correct_down":       correct_down,
        "direction_accuracy": dir_acc,
    })

    # 7) Compute 4-step auto-regressive trend accuracy
    total_w4 = 0
    correct_w4 = 0

    model.eval()
    with torch.no_grad():
        # iterate each CSV in TEST_CSV_DIR
        for fn in os.listdir(TEST_CSV_DIR):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(TEST_CSV_DIR, fn)
            df = pd.read_csv(path)
            tokens = df["Token"].tolist()
            # for each possible window
            for i in range(len(tokens) - CONTEXT_SIZE - 3):
                total_w4 += 1
                window = tokens[i : i + CONTEXT_SIZE]
                # autoregressively predict next 4
                pred_token = None
                for _ in range(4):
                    inp = torch.tensor([window], dtype=torch.long, device=DEVICE)  # [1, CONTEXT_SIZE]
                    logits = model(inp)
                    pred_token = logits.argmax(dim=1).item()
                    # slide window
                    window = window[1:] + [pred_token]

                # compare direction of 4th pred vs true 4th ahead
                true_token4 = tokens[i + CONTEXT_SIZE + 3]
                o_t4, c_t4 = vocab_map[str(true_token4)][0], vocab_map[str(true_token4)][3]
                o_p4, c_p4 = vocab_map[str(pred_token)][0], vocab_map[str(pred_token)][3]
                dir_true4 = int(c_t4 >= o_t4)
                dir_pred4 = int(c_p4 >= o_p4)
                if dir_true4 == dir_pred4:
                    correct_w4 += 1

    metrics.update({
        "total_windows_4step": total_w4,
        "correct_trend4":      correct_w4,
        "trend4_accuracy":     correct_w4 / total_w4 if total_w4 else 0.0
    })

    # 8) Save all metrics
    json_path = os.path.join(LOGS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    txt_path = os.path.join(LOGS_DIR, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Inference Metrics\n=================\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"All metrics saved to {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
