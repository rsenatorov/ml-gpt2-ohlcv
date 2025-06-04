#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
eval_trade_sim.py – Confidence vs Coverage vs Direction vs PnL simulation
using real price data and pure direction signals.
Trades full bank each signal: long if model predicts up, short if down,
exits at the real close of the same candle.
Outputs CSV of coverage,confidence,direction_acc,total_profit,avg_gain,avg_loss
and saves 4 sample trade plots (2 wins, 2 losses) in logs/test.
"""
import os
import csv
import json
import random

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from network.model import GPT2TimeSeries

# ── CONFIG ────────────────────────────────────────────────────────────
PAIR         = "AAPLUSUSD"
TF           = "M5"
CONTEXT_SIZE = 100
LAST_CANDLES = 24         # history + predicted candle to show
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIN_WIDTH    = 0.05
NUM_BINS     = int(1.0 / BIN_WIDTH)
INITIAL_BANK = 1000.0     # starting capital

# Paths
PRICE_CSV = os.path.join("data", "test", "market_data", f"{PAIR}_{TF}.csv")
TOK_CSV   = os.path.join("data", "test", "tokens",      f"TOKN_{PAIR}_{TF}.csv")
VOCAB     = os.path.join("models", "vocab.json")
CKPT      = os.path.join("checkpoints", "checkpoint3.pth")
LOGS_DIR  = os.path.join("logs", "test")
OUT_CSV   = os.path.join(LOGS_DIR, "trade_confidence3.csv")
# ───────────────────────────────────────────────────────────────────────

def draw_candles(ax, o, h, l, c, highlight_idx):
    """Draw OHLC candlesticks; bar at highlight_idx is thicker/colored."""
    x = np.arange(len(o))
    for i in range(len(o)):
        up = c[i] >= o[i]
        color = "tab:green" if up else "tab:red"
        lw = 2.0 if i == highlight_idx else 0.7
        wick_color = color if i == highlight_idx else "black"
        ax.vlines(x[i], l[i], h[i], color=wick_color, linewidth=lw, zorder=2)
        bottom = min(o[i], c[i])
        height = abs(c[i] - o[i]) or 1e-9
        rect = Rectangle(
            (x[i] - 0.3, bottom), 0.6, height,
            facecolor=color, edgecolor=wick_color,
            linewidth=lw, zorder=3
        )
        ax.add_patch(rect)
    ax.set_xlim(-1, len(o))
    ax.set_xticks([])
    ax.set_ylabel("Price")

def plot_trade_sample(df, trade, filename):
    """
    Plot the last LAST_CANDLES+1 bars for a trade,
    highlight the predicted bar, show entry & exit lines,
    title with ROI %.
    """
    start = trade["start_idx"]
    window = CONTEXT_SIZE + 1
    seg = df.iloc[start:start+window].reset_index(drop=True)
    seg = seg.iloc[-(LAST_CANDLES+1):].reset_index(drop=True)

    o = seg["Open"].to_numpy()
    h = seg["High"].to_numpy()
    l = seg["Low"].to_numpy()
    c = seg["Close"].to_numpy()

    entry = trade["entry"]
    exitp = trade["exit"]
    roi   = trade["return_pct"]
    is_long = trade["is_long"]

    fig, ax = plt.subplots(figsize=(10,6))
    hi = len(o)-1
    # background highlight
    ax.axvspan(hi-0.5, hi+0.5, color="yellow", alpha=0.3, zorder=1)
    draw_candles(ax, o, h, l, c, highlight_idx=hi)
    # entry & exit lines
    ax.hlines(entry, 0, hi, linestyle="--", label="Entry")
    ax.hlines(exitp,  0, hi, linestyle=":",  label="Exit")
    direction = "Long" if is_long else "Short"
    ax.set_title(
        f"{PAIR} {TF}  idx={start}   {direction} ROI: {roi:.2%}",
        fontweight="bold"
    )
    ax.legend(loc="upper left")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close(fig)

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    # load & merge
    df_price = pd.read_csv(PRICE_CSV, parse_dates=["Time"])
    df_tok   = pd.read_csv(TOK_CSV,   parse_dates=["Time"])
    df = (pd.merge(df_tok, df_price, on="Time", how="inner")
            .sort_values("Time").reset_index(drop=True))
    dropped = len(df_tok) - len(df)
    if dropped:
        print(f"Warning: dropped {dropped} token rows without prices")

    # tokens
    tokens = df["Token"].astype(int).tolist()
    total = len(tokens) - CONTEXT_SIZE

    # load model + vocab
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    vocab_size = state["lm_head.weight"].shape[0]
    model = GPT2TimeSeries(vocab_size=vocab_size, context_size=CONTEXT_SIZE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    with open(VOCAB) as f:
        vocab_map = json.load(f)

    confidences, dir_corr, returns = [], [], []
    trades = []

    with torch.no_grad():
        for i in range(total):
            ctx = torch.tensor([tokens[i:i+CONTEXT_SIZE]],
                               dtype=torch.long, device=DEVICE)
            logits = model(ctx)
            if logits.dim()==3:
                logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            prob, pred = probs.max(dim=-1)
            prob, pred = prob.item(), pred.item()

            j = i + CONTEXT_SIZE
            if j >= len(df):
                continue

            row = df.iloc[j]
            prev= df.iloc[j-1]
            o_real = row["Open"]
            c_real = row["Close"]

            # predicted normalized prices
            o_n, _, _, c_n, _ = vocab_map[str(pred)]
            is_long = (c_n >= o_n)

            # simple return
            if is_long:
                ret = (c_real - o_real) / o_real
            else:
                ret = (o_real - c_real) / o_real

            # true direction correct?
            true_tok = tokens[j]
            o_t, _, _, c_t, _ = vocab_map[str(true_tok)]
            dir_ok = int((c_n>=o_n) == (c_t>=o_t))

            confidences.append(prob)
            dir_corr.append(dir_ok)
            returns.append(ret)
            trades.append({
                "start_idx": i,
                "is_long":   is_long,
                "entry":     o_real,
                "exit":      c_real,
                "return_pct":ret
            })

    total_trades = len(confidences)

    # write CSV by confidence bin
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage","confidence","direction_acc",
                    "total_profit","avg_gain","avg_loss"])
        for b in range(NUM_BINS):
            thr = b * BIN_WIDTH
            idx = [k for k,c in enumerate(confidences) if c>=thr]
            cnt = len(idx)
            cov = cnt/total_trades if total_trades else 0.0
            da  = sum(dir_corr[k] for k in idx)/cnt if cnt else 0.0
            gains  = [returns[k] for k in idx if returns[k]>0]
            losses = [returns[k] for k in idx if returns[k]<0]
            ag = sum(gains)/len(gains) if gains else 0.0
            al = sum(losses)/len(losses) if losses else 0.0

            bank = INITIAL_BANK
            for k in idx:
                bank *= (1 + returns[k])
            tp = bank - INITIAL_BANK

            w.writerow([f"{cov:.6f}",f"{thr:.2f}",
                        f"{da:.6f}",f"{tp:.6f}",
                        f"{ag:.6f}",f"{al:.6f}"])

    # print summary
    print("\nThr │ Cov%   │ Dir%   │ Total PnL │ AvgGain │ AvgLoss")
    print("───┼────────┼────────┼───────────┼─────────┼─────────")
    with open(OUT_CSV) as f:
        next(f)
        for line in f:
            cov,thr,da,tp,ag,al = line.strip().split(",")
            print(f"≥{float(thr):.2f} │ {float(cov):>6.2%} │ "
                  f"{float(da):>6.2%} │ {float(tp):>9.2f} │ "
                  f"{float(ag):>7.4f} │ {float(al):>7.4f}")

    # pick 2 winners & 2 losers
    wins = [t for t in trades if t["return_pct"]>0]
    los = [t for t in trades if t["return_pct"]<0]
    samp = []
    samp += random.sample(wins, min(2,len(wins)))
    samp += random.sample(los,  min(2,len(los)))

    # save 4 plots
    for idx, tr in enumerate(samp, 1):
        fn = os.path.join(LOGS_DIR, f"sample_{idx}.png")
        plot_trade_sample(df, tr, fn)
    print(f"\nSaved {len(samp)} sample plots to {LOGS_DIR}/")

if __name__ == "__main__":
    main()