#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
normalize_market_data.py

Stochastic-normalise each OHLCV CSV in data/market_data over a backward-looking
window of the last 100 candles. Saves a Joblib “RowStochasticNormalizer” so that
you can load & apply the exact same logic in inference. Outputs NORM_{pair_tf}.csv
into data/norm/, and writes the normalizer (including its output column list)
to models/stochastic_normalizer.joblib.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

# ── CONFIG ─────────────────────────────────────────────────────────────
INPUT_DIR    = os.path.join("data", "market_data")
OUTPUT_DIR   = os.path.join("data", "norm")
MODEL_DIR    = "models"
WINDOW_SIZE  = 100
EPS          = 1e-9
COLUMNS      = [
    "Time", "Open", "High", "Low", "Close", "Volume",
    "Low_min", "High_max", "Vol_max"
]
# ────────────────────────────────────────────────────────────────────────

class RowStochasticNormalizer:
    """Stochastic min-max normaliser with a fixed lookback window."""
    def __init__(self, window_size: int, eps: float = 1e-9):
        self.window_size = window_size
        self.eps = eps
        # include the full output column list in the object
        self.columns = COLUMNS.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # compute rolling min/max over the past WINDOW_SIZE bars
        low_min  = df["Low"].rolling(window=self.window_size, min_periods=self.window_size).min()
        high_max = df["High"].rolling(window=self.window_size, min_periods=self.window_size).max()
        vol_max  = df["Volume"].rolling(window=self.window_size, min_periods=self.window_size).max()

        df2 = df.copy()
        df2["Low_min"]  = low_min.values
        df2["High_max"] = high_max.values
        df2["Vol_max"]  = vol_max.values

        # drop rows that couldn't form a full window
        df2 = df2.dropna(subset=["Low_min", "High_max", "Vol_max"]).reset_index(drop=True)

        # apply per-row normalization
        price_range = (df2["High_max"] - df2["Low_min"]).clip(lower=self.eps)
        for col in ["Open", "High", "Low", "Close"]:
            df2[col] = (df2[col] - df2["Low_min"]) / price_range
        df2["Volume"] = df2["Volume"] / df2["Vol_max"].clip(lower=self.eps)

        # return exactly the nine columns stored in self.columns
        return df2[self.columns]


def _load_and_prune(path: str) -> pd.DataFrame:
    """
    Manually parse TSV/space-delimited file, drop everything up to+including
    the first bad row, then return a clean DataFrame with only the required
    six columns.
    """
    raw_rows = []
    first_bad = False
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 7:
                if not first_bad:
                    first_bad = True
                    raw_rows = []
                continue
            ts = parts[0] + " " + parts[1]
            try:
                o, h, l, c, v = map(float, parts[2:7])
            except:
                if not first_bad:
                    first_bad = True
                    raw_rows = []
                continue
            raw_rows.append({
                "Time":   pd.to_datetime(ts),
                "Open":   o,
                "High":   h,
                "Low":    l,
                "Close":  c,
                "Volume": v
            })

    if not raw_rows:
        return pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(raw_rows)
    return (
        df.drop_duplicates(subset="Time")
          .sort_values("Time")
          .reset_index(drop=True)
    )


def normalize_file(path: str, normalizer: RowStochasticNormalizer) -> None:
    filename = os.path.basename(path)
    df = _load_and_prune(path)
    if df.empty:
        print(f"Skipping {filename}: no valid data after pruning")
        return

    if len(df) <= WINDOW_SIZE:
        print(f"Skipping {filename}: only {len(df)} rows (≤ {WINDOW_SIZE})")
        return

    df_norm = normalizer.transform(df)
    if df_norm.empty:
        print(f"Skipping {filename}: no rows left after normalization")
        return

    out_name = f"NORM_{os.path.splitext(filename)[0]}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    df_norm.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved {out_path} ({len(df_norm)} rows)")


def main() -> None:
    # instantiate and save normalizer (now including its .columns list)
    normalizer = RowStochasticNormalizer(WINDOW_SIZE, EPS)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib_path = os.path.join(MODEL_DIR, "stochastic_normalizer.joblib")
    joblib.dump(normalizer, joblib_path)
    print(f"Saved normalizer (with output column spec) to {joblib_path}")

    # process each CSV
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    for path in tqdm(files, desc="Normalizing files"):
        try:
            normalize_file(path, normalizer)
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}: {e}")


if __name__ == "__main__":
    main()
