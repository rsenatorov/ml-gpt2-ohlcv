#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
sanity_check3.py – compare real vs token‐reconstructed price (per‐bar anchored)

For a given instrument/timeframe, pick one WINDOW‐size window and plot:
  1) Real OHLC (reconstructed from normalized+min/max)
  2) Token→decoded OHLC, then un‐normalized and chained so each open
     = previous bar’s decoded close

Usage:
    python sanity_check3.py [--pair EURUSD] [--tf D1]
                            [--window 100] [--seed 42]
"""
import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── DEFAULTS ────────────────────────────────────────────────────────────
PAIR     = "AAPLUSUSD"    # e.g. BTCUSDT, ETHUSD, etc.
TF       = "D1"         # e.g. 1h, 4h, D1, M1
WINDOW   = 100          # bars per window
SEED     = None         # for reproducibility

NORM_DIR   = os.path.join("data/test", "norm")
TOK_DIR    = os.path.join("data/test", "tokens")
VOCAB_PATH = os.path.join("models", "vocab.json")
OUT_DIR    = "data/test"
# ───────────────────────────────────────────────────────────────────────

def draw_candles(ax, o, h, l, c, title):
    """Draw OHLC candlesticks (no volume) on ax."""
    x = np.arange(len(o))
    for i in range(len(o)):
        up = c[i] >= o[i]
        ax.vlines(x[i], l[i], h[i], color="black", linewidth=0.7)
        bot    = min(o[i], c[i])
        height = abs(c[i] - o[i]) or 1e-9
        rect   = Rectangle((x[i]-0.3, bot), 0.6, height,
                           color="tab:green" if up else "tab:red")
        ax.add_patch(rect)
    ax.set_xlim(-1, len(o))
    ax.set_xticks([])
    ax.set_ylabel("Price")
    ax.set_title(title, fontweight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair",   type=str, default=PAIR,
                   help="Instrument code, e.g. EURUSD")
    p.add_argument("--tf",     type=str, default=TF,
                   help="Timeframe, e.g. D1, 1h")
    p.add_argument("--window", type=int, default=WINDOW,
                   help="Bars per window")
    p.add_argument("--seed",   type=int, default=SEED,
                   help="Random seed")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # file paths
    norm_fp  = os.path.join(NORM_DIR, f"NORM_{args.pair}_{args.tf}.csv")
    token_fp = os.path.join(TOK_DIR,  f"TOKN_{args.pair}_{args.tf}.csv")
    for fp in (norm_fp, token_fp, VOCAB_PATH):
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing: {fp}")

    # load CSVs
    df_norm   = pd.read_csv(norm_fp, parse_dates=["Time"])
    df_tokens = pd.read_csv(token_fp, parse_dates=["Time"])
    if len(df_norm) != len(df_tokens):
        raise ValueError("Row count mismatch between norm & token CSVs")

    # ensure min/max columns exist
    for col in ("Low_min","High_max","Vol_max"):
        if col not in df_norm:
            raise ValueError(f"{norm_fp} missing column {col}")

    total = len(df_norm)
    if total < args.window:
        raise ValueError(f"Need ≥{args.window} rows, got {total}")

    # load vocab mapping: token → [O,H,L,C,V]
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    # pick a random window
    start = random.randrange(0, total - args.window + 1)
    end   = start + args.window
    norm_win  = df_norm.iloc[start:end].reset_index(drop=True)
    token_win = df_tokens.iloc[start:end].reset_index(drop=True)

    # 1) reconstruct the **real** OHLC from normalized + min/max
    low_min   = norm_win["Low_min"].to_numpy()
    high_max  = norm_win["High_max"].to_numpy()
    scale     = high_max - low_min

    o_real = norm_win["Open"].to_numpy()   * scale + low_min
    h_real = norm_win["High"].to_numpy()  * scale + low_min
    l_real = norm_win["Low"].to_numpy()   * scale + low_min
    c_real = norm_win["Close"].to_numpy() * scale + low_min

    # 2) decode each token → normalized OHLC → un-normalize
    codes   = token_win["Token"].astype(int).tolist()
    decoded = np.array([vocab[str(code)] for code in codes], dtype=np.float32)
    o_dec_n, h_dec_n, l_dec_n, c_dec_n, _ = decoded.T

    o_dec = o_dec_n * scale + low_min
    h_dec = h_dec_n * scale + low_min
    l_dec = l_dec_n * scale + low_min
    c_dec = c_dec_n * scale + low_min

    # 3) compute the “prior close” in real prices
    if start > 0:
        prev = df_norm.iloc[start-1]
        prev_scale = prev["High_max"] - prev["Low_min"]
        prior_close = prev["Close"] * prev_scale + prev["Low_min"]
    else:
        # no prior bar: use the real open of first bar
        prior_close = o_real[0]

    # 4) build a chained, anchored decoded series
    aligned_o = np.zeros_like(o_dec)
    aligned_h = np.zeros_like(h_dec)
    aligned_l = np.zeros_like(l_dec)
    aligned_c = np.zeros_like(c_dec)

    # precompute each bar’s intra-bar offsets
    dh = h_dec - o_dec
    dl = l_dec - o_dec
    dc = c_dec - o_dec

    # bar 0 anchored
    aligned_o[0] = prior_close
    aligned_h[0] = aligned_o[0] + dh[0]
    aligned_l[0] = aligned_o[0] + dl[0]
    aligned_c[0] = aligned_o[0] + dc[0]

    # each subsequent bar’s open = previous bar’s close
    for i in range(1, args.window):
        aligned_o[i] = aligned_c[i-1]
        aligned_h[i] = aligned_o[i] + dh[i]
        aligned_l[i] = aligned_o[i] + dl[i]
        aligned_c[i] = aligned_o[i] + dc[i]

    # 5) plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios":[1,1]}
    )
    draw_candles(
        ax1, o_real, h_real, l_real, c_real,
        f"{args.pair} {args.tf}  Real price  (rows {start}–{end-1})"
    )
    draw_candles(
        ax2, aligned_o, aligned_h, aligned_l, aligned_c,
        f"{args.pair} {args.tf}  Token→Decoded (per-bar anchored)"
    )

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUT_DIR,
        f"sanity3_{args.pair}_{args.tf}_start{start}.png"
    )
    plt.savefig(out_path, dpi=140)
    print(f"Saved comparison plot → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
