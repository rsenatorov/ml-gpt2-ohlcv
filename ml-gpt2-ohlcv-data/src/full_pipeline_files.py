#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
full_pipeline_local.py – Load, sanitize, normalize, and tokenize local AAPL/USDT daily data

Pipeline steps:
 1. Load and prune the local TSV/CSV from data/test/market_data/AAPLUSUSD_D1.csv
    (handles the extra “volumeto” column exactly as normalize_market_data.py does)
 2. Drop any leading rows with missing OHLCV values
 3. Save cleaned market data back to data/test/market_data/AAPLUSUSD_D1.csv
 4. Load saved RowStochasticNormalizer and normalize (window=100),
    saving NORM_AAPLUSUSD_D1.csv into data/test/norm
 5. Load VQ-VAE encoder+codebook and tokenize each row,
    saving TOKN_AAPLUSUSD_D1.csv into data/test/tokens
"""
import os
import re
import pandas as pd
import joblib
import torch
from torch import nn
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_DIR       = "data/test"
MARKET_DIR     = os.path.join(BASE_DIR, "market_data")
INPUT_FILE     = os.path.join(MARKET_DIR, "AAPLUSUSD_M5.csv")
NORM_DIR       = os.path.join(BASE_DIR, "norm")
TOKENS_DIR     = os.path.join(BASE_DIR, "tokens")

MODEL_DIR      = "models"
JOBLIB_PATH    = os.path.join(MODEL_DIR, "stochastic_normalizer.joblib")
VQ_MODEL_PATH  = os.path.join(MODEL_DIR, "ohlcv_vqvae_encoder.pth")

WINDOW_SIZE    = 100
EPS            = 1e-9
FEATURE_COLS   = ["Open", "High", "Low", "Close", "Volume"]
TIMEFRAME      = "M5"

# VQ-VAE hyperparams (must match your training!)
K_CODES        = 2048
LATENT_D       = 128
BETA           = 0.25
DECAY_GAMMA    = 0.99
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ensure directories exist
for d in (MARKET_DIR, NORM_DIR, TOKENS_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)


def _load_and_prune(path: str) -> pd.DataFrame:
    """
    Parse whitespace-delimited rows, drop everything up to+including
    the first bad row, then return a clean DataFrame with only six columns:
    Time, Open, High, Low, Close, Volume.
    """
    raw = []
    first_bad = False
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 7:
                if not first_bad:
                    first_bad = True
                    raw = []
                continue
            ts = parts[0] + " " + parts[1]
            try:
                o, h, l, c, v = map(float, parts[2:7])
            except:
                if not first_bad:
                    first_bad = True
                    raw = []
                continue
            raw.append({
                "Time":   pd.to_datetime(ts),
                "Open":   o,
                "High":   h,
                "Low":    l,
                "Close":  c,
                "Volume": v,
            })

    if not raw:
        return pd.DataFrame(columns=["Time"] + FEATURE_COLS)

    df = pd.DataFrame(raw)
    return (
        df.drop_duplicates(subset="Time")
          .sort_values("Time")
          .reset_index(drop=True)
    )


def drop_leading_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all leading rows that have any NaN in Open/High/Low/Close/Volume.
    """
    df = df.reset_index(drop=True)
    mask = df[FEATURE_COLS].isnull().any(axis=1)
    if mask.any():
        df = df.iloc[mask.idxmax() + 1 :].reset_index(drop=True)
    return df


class RowStochasticNormalizer:
    """Same as in normalize_market_data.py."""
    def __init__(self, window_size: int, eps: float = 1e-9):
        self.window_size = window_size
        self.eps = eps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        low_min  = df["Low"].rolling(window=self.window_size, min_periods=self.window_size).min()
        high_max = df["High"].rolling(window=self.window_size, min_periods=self.window_size).max()
        vol_max  = df["Volume"].rolling(window=self.window_size, min_periods=self.window_size).max()

        df2 = df.copy()
        df2["Low_min"]  = low_min.values
        df2["High_max"] = high_max.values
        df2["Vol_max"]  = vol_max.values

        df2 = df2.dropna(subset=["Low_min", "High_max", "Vol_max"]).reset_index(drop=True)

        price_range = (df2["High_max"] - df2["Low_min"]).clip(lower=self.eps)
        for col in ("Open", "High", "Low", "Close"):
            df2[col] = (df2[col] - df2["Low_min"]) / price_range
        df2["Volume"] = df2["Volume"] / df2["Vol_max"].clip(lower=self.eps)

        return df2[[
            "Time", "Open", "High", "Low",
            "Close", "Volume",
            "Low_min", "High_max", "Vol_max"
        ]]


class Encoder(nn.Module):
    """Must match your trained encoder architecture."""
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_latent),
        )
    def forward(self, x): return self.net(x)


def main():
    # 1. Load & prune local CSV
    df_raw = _load_and_prune(INPUT_FILE)
    if df_raw.empty:
        print("No valid rows after pruning; exiting.")
        return

    # 2. Drop leading NaNs only
    df_clean = drop_leading_nans(df_raw)
    if df_clean.empty:
        print("No data left after dropping leading NaNs; exiting.")
        return

    # 3. Save cleaned market data (overwrite)
    out_mkt = df_clean.copy()
    out_mkt["Time"] = out_mkt["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_mkt.to_csv(INPUT_FILE, index=False, float_format="%.6f")
    print(f"✔ Market data saved: {INPUT_FILE}")

    # 4. Normalize
    if not os.path.exists(JOBLIB_PATH):
        raise FileNotFoundError(f"Missing normalizer at {JOBLIB_PATH}")
    normalizer = joblib.load(JOBLIB_PATH)

    df_norm = normalizer.transform(df_clean)
    out_norm = df_norm.copy()
    out_norm["Time"] = out_norm["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    norm_path = os.path.join(NORM_DIR, f"NORM_AAPLUSUSD_{TIMEFRAME}.csv")
    out_norm.to_csv(norm_path, index=False, float_format="%.6f")
    print(f"✔ Normalized data saved: {norm_path}")

    # 5. Tokenize via VQ-VAE
    if not os.path.exists(VQ_MODEL_PATH):
        raise FileNotFoundError(f"Missing VQ-VAE model at {VQ_MODEL_PATH}")
    chkpt = torch.load(VQ_MODEL_PATH, map_location=DEVICE)

    enc = Encoder(in_dim=5, d_latent=LATENT_D).to(DEVICE)
    vq  = VectorQuantize(
        dim=LATENT_D,
        codebook_size=K_CODES,
        decay=DECAY_GAMMA,
        kmeans_init=False,
        commitment_weight=BETA
    ).to(DEVICE)

    enc.load_state_dict(chkpt["encoder_state"])
    vq.load_state_dict(chkpt["vq_state"])
    enc.eval();  vq.eval()

    data_tensor = torch.tensor(
        df_norm[FEATURE_COLS].values,
        dtype=torch.float32,
        device=DEVICE
    )
    with torch.no_grad():
        out = vq(enc(data_tensor))
        codes = out[1] if isinstance(out, tuple) else vq(enc(data_tensor))[1]

    toks = codes.cpu().numpy()
    df_tok = pd.DataFrame({
        "Time":  out_norm["Time"],
        "Token": toks
    })
    tok_path = os.path.join(TOKENS_DIR, f"TOKN_AAPLUSUSD_{TIMEFRAME}.csv")
    df_tok.to_csv(tok_path, index=False)
    print(f"✔ Tokens saved: {tok_path}")


if __name__ == "__main__":
    main()
