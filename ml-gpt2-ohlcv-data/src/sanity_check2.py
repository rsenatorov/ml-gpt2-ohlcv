#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
sanity_check2.py – compare normalized vs token-decoded OHLCV candles

Edit the CONFIG section below to choose which pair, timeframe, number of windows,
and random seed to use. Then simply run:

    python sanity_check2.py

It will read:
  • data/norm/NORM_{PAIR}_{TF}.csv
  • data/tokens/TOKN_{PAIR}_{TF}.csv

and produce N sample plots of 100-candle windows, saved as:
  data/sanity2_{PAIR}_{TF}_start{idx}.png
"""

import os
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
PAIR         = "EURUSD"      # e.g. "BTCUSDT", "ETHUSD", etc.
TF           = "D1"           # timeframe, e.g. "1h", "4h", "D1"
SAMPLES      = 3              # how many random windows to plot
SEED         = 42             # set to None for no fixed seed
WINDOW_SIZE  = 100            # number of candles per window

NORM_DIR     = os.path.join("data", "norm")
TOK_DIR      = os.path.join("data", "tokens")
VOCAB_PATH   = os.path.join("models", "vocab.json")
OUT_DIR      = "data"         # where to save sanity2_*.png
# ───────────────────────────────────────────────────────────────────────

def plot_window(norm_win, token_win, vocab, start_idx):
    """Plot one WINDOW_SIZE-long window of normalized vs decoded-token candles."""
    # normalized arrays
    o_n = norm_win["Open"].to_numpy()
    h_n = norm_win["High"].to_numpy()
    l_n = norm_win["Low"].to_numpy()
    c_n = norm_win["Close"].to_numpy()
    v_n = norm_win["Volume"].to_numpy()

    # decode tokens → floats
    codes = token_win["Token"].astype(int).tolist()
    decoded = np.array([vocab[str(code)] for code in codes], dtype=np.float32)
    o_t, h_t, l_t, c_t, v_t = decoded.T

    # set up figure
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    x = np.arange(WINDOW_SIZE)

    # draw candles
    for i in range(WINDOW_SIZE):
        col_n = "green" if (i == 0 or c_n[i] >= c_n[i-1]) else "red"
        col_t = "green" if (i == 0 or c_t[i] >= c_t[i-1]) else "red"

        # normalized (solid)
        ax_p.vlines(x[i] - 0.2, l_n[i], h_n[i], color=col_n, linewidth=0.7)
        ax_p.add_patch(Rectangle(
            (x[i] - 0.4, min(o_n[i], c_n[i])),
            0.4, abs(c_n[i] - o_n[i]) + 1e-9,
            color=col_n
        ))
        # token-decoded (transparent)
        ax_p.vlines(x[i] + 0.2, l_t[i], h_t[i], color=col_t, linewidth=0.7, alpha=0.5)
        ax_p.add_patch(Rectangle(
            (x[i] + 0.0, min(o_t[i], c_t[i])),
            0.4, abs(c_t[i] - o_t[i]) + 1e-9,
            color=col_t, alpha=0.5
        ))

    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_ylabel("Price (normalized)")
    ax_p.set_title(f"{PAIR} {TF}  window start={start_idx}", fontweight="bold")

    # draw volumes
    ax_v.bar(x - 0.2, v_n, width=0.4, label="norm", alpha=0.7)
    ax_v.bar(x + 0.2, v_t, width=0.4, label="token", alpha=0.5)
    ax_v.set_ylim(0, 1.05)
    ax_v.set_ylabel("Volume")
    ax_v.set_xlabel("Bars (oldest → newest)")
    ax_v.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"sanity2_{PAIR}_{TF}_start{start_idx}.png")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot → {out_path}")

def main():
    # optional seeding
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    # input paths
    norm_path  = os.path.join(NORM_DIR,  f"NORM_{PAIR}_{TF}.csv")
    token_path = os.path.join(TOK_DIR,   f"TOKN_{PAIR}_{TF}.csv")

    # sanity checks
    for fp in (norm_path, token_path, VOCAB_PATH):
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing: {fp}")

    # load data
    df_norm   = pd.read_csv(norm_path, parse_dates=["Time"])
    df_tokens = pd.read_csv(token_path, parse_dates=["Time"])
    if len(df_norm) != len(df_tokens):
        raise ValueError("Mismatch in row counts between norm & token CSVs")

    total = len(df_norm)
    if total < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} rows, got {total}")

    # load vocab
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    # pick random window starts
    os.makedirs(OUT_DIR, exist_ok=True)
    starts = random.sample(range(0, total - WINDOW_SIZE + 1), SAMPLES)

    for start in tqdm(starts, desc="Plotting windows"):
        norm_win  = df_norm.iloc[start : start + WINDOW_SIZE]
        token_win = df_tokens.iloc[start : start + WINDOW_SIZE]
        plot_window(norm_win, token_win, vocab, start)

if __name__ == "__main__":
    main()
