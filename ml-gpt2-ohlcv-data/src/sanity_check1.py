#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
sanity_check1.py

Compare two 100-candle windows from the first raw CSV in data/market_data
against its normalized version in data/norm (which now uses backward-looking
100-bar normalization, so the norm file begins at raw row WINDOW_SIZE-1).
Draws proper candlestick bodies, wicks, and volume bars.

Usage:
    python src/sanity_check1.py
    # or specify a different file pair:
    python src/sanity_check1.py --pair ADAUSDT --tf D1
"""

import os
import glob
import re
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── CONFIG ─────────────────────────────────────────────────────────────
RAW_DIR   = "data/market_data"
NORM_DIR  = "data/norm"
WINDOW    = 100
OFFSET    = WINDOW - 1   # first norm row corresponds to raw row WINDOW-1
# ────────────────────────────────────────────────────────────────────────

def load_and_prune(path: str) -> pd.DataFrame:
    """Manual loader: whitespace-split, drop up to first bad row, return DF."""
    raws = []
    first_bad = False
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 7:
                if not first_bad:
                    first_bad = True
                    raws = []
                continue
            ts = parts[0] + " " + parts[1]
            try:
                o, h, l, c, v = map(float, parts[2:7])
            except:
                if not first_bad:
                    first_bad = True
                    raws = []
                continue
            raws.append({
                "Time":   pd.to_datetime(ts),
                "Open":   o,
                "High":   h,
                "Low":    l,
                "Close":  c,
                "Volume": v,
            })
    df = pd.DataFrame(raws)
    if df.empty:
        return df
    return df.drop_duplicates("Time").sort_values("Time").reset_index(drop=True)

def draw_candles(ax, o, h, l, c, volume=None):
    """
    Draw candlestick wicks and bodies on ax.
    If `volume` is provided, also plot a twin-axis bar chart of volume.
    """
    x = np.arange(len(o))
    # set up volume axis if needed
    ax_vol = None
    if volume is not None:
        ax_vol = ax.twinx()
        max_vol = volume.max()
        ax_vol.set_ylim(0, max_vol * 1.1)
        ax_vol.set_ylabel("Volume", color="gray")
        ax_vol.bar(x, volume, width=0.6, alpha=0.3, color="gray")

    for i in range(len(o)):
        up = c[i] >= o[i]
        # wick
        ax.vlines(x[i], l[i], h[i], color="black", linewidth=0.7)
        # body
        bottom = min(o[i], c[i])
        height = abs(c[i] - o[i])
        rect = Rectangle(
            (x[i] - 0.3, bottom),
            0.6, height if height>0 else 1e-6,
            color="tab:green" if up else "tab:red"
        )
        ax.add_patch(rect)

    ax.set_ylabel("Price")
    ax.set_xlim(-1, len(o))
    ax.set_xticks([])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=str, default=None,
                   help="Instrument code, e.g. ADAUSDT")
    p.add_argument("--tf",   type=str, default=None,
                   help="Timeframe suffix, e.g. D1 or M1")
    args = p.parse_args()

    # pick files
    if args.pair and args.tf:
        raw_glob  = os.path.join(RAW_DIR,  f"{args.pair}_{args.tf}.csv")
        norm_glob = os.path.join(NORM_DIR, f"NORM_{args.pair}_{args.tf}.csv")
    else:
        raw_glob  = os.path.join(RAW_DIR,  "*.csv")
        norm_glob = os.path.join(NORM_DIR, "NORM_*.csv")

    raws  = sorted(glob.glob(raw_glob))
    norms = sorted(glob.glob(norm_glob))
    if not raws or not norms:
        raise FileNotFoundError("Could not find matching raw/norm files.")

    raw_path  = raws[0]
    norm_path = norms[0]

    # load data
    df_raw  = load_and_prune(raw_path)
    df_norm = pd.read_csv(norm_path, parse_dates=["Time"])

    # sanity length
    needed_raw = OFFSET + 2*WINDOW
    if len(df_raw) < needed_raw or len(df_norm) < 2*WINDOW:
        raise ValueError(f"Need ≥ {needed_raw} raw rows and ≥ {2*WINDOW} norm rows.")

    # slice windows
    w1_raw  = df_raw.iloc[OFFSET         : OFFSET + WINDOW].reset_index(drop=True)
    w1_norm = df_norm.iloc[0             : WINDOW         ].reset_index(drop=True)
    w2_raw  = df_raw.iloc[OFFSET + WINDOW : OFFSET + 2*WINDOW].reset_index(drop=True)
    w2_norm = df_norm.iloc[WINDOW        : 2*WINDOW        ].reset_index(drop=True)

    # build 4 subplots: price & volume for window1 and window2, raw vs norm
    fig, axes = plt.subplots(
        4, 2, figsize=(14, 16),
        gridspec_kw={"height_ratios": [3,1,3,1]},
        sharex="col"
    )
    ax1, ax2    = axes[0]
    ax_vol1, ax_vol2 = axes[1]
    ax3, ax4    = axes[2]
    ax_vol3, ax_vol4 = axes[3]

    # Window 1 price
    draw_candles(ax1,
                 w1_raw["Open"], w1_raw["High"],
                 w1_raw["Low"],  w1_raw["Close"],
                 volume=w1_raw["Volume"])
    ax1.set_title(f"Raw   Window 1 (rows {OFFSET}–{OFFSET+WINDOW-1})")
    draw_candles(ax2,
                 w1_norm["Open"], w1_norm["High"],
                 w1_norm["Low"],  w1_norm["Close"],
                 volume=w1_norm["Volume"])
    ax2.set_title("Norm. Window 1")

    # Window 2 price
    draw_candles(ax3,
                 w2_raw["Open"], w2_raw["High"],
                 w2_raw["Low"],  w2_raw["Close"],
                 volume=w2_raw["Volume"])
    ax3.set_title(f"Raw   Window 2 (rows {OFFSET+WINDOW}–{OFFSET+2*WINDOW-1})")
    draw_candles(ax4,
                 w2_norm["Open"], w2_norm["High"],
                 w2_norm["Low"],  w2_norm["Close"],
                 volume=w2_norm["Volume"])
    ax4.set_title("Norm. Window 2")

    # label x-axes
    for ax in (ax_vol1, ax_vol2, ax_vol3, ax_vol4):
        ax.set_xlabel("Bar index")
        ax.set_xlim(-1, WINDOW)
    ax_vol1.set_ylabel("Volume")
    ax_vol2.set_ylabel("Volume")
    ax_vol3.set_ylabel("Volume")
    ax_vol4.set_ylabel("Volume")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
