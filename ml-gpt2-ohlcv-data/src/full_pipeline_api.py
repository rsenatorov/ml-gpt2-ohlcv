#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
full_pipeline.py - Fetch, sanitize, normalize, and tokenize WAVES/USDT daily data

Pipeline steps:
 1. Fetch OHLCV from CryptoCompare histoday API
 2. Sanity-check: drop any preamble with missing values or time gaps
 3. Save cleaned market data to data/test/market_data/WAVESUSDT_D1.csv
 4. Load saved stochastic normalizer (window=100) and normalize data (backward-looking),
    saving all nine columns into data/test/norm/NORM_WAVESUSDT_D1.csv
 5. Load VQ-VAE encoder+codebook and tokenize each row
 6. Save tokens to data/test/tokens/TOKN_WAVESUSDT_D1.csv
"""
import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# ── CONFIG ────────────────────────────────────────────────────────────
PAIR           = "WAVES/USDT"
FSYM, TSYM     = PAIR.split("/")
TIMEFRAME      = "D1"
API_URL        = "https://min-api.cryptocompare.com/data/v2/histoday"

BASE_DIR       = "data/test"
MARKET_DIR     = os.path.join(BASE_DIR, "market_data")
NORM_DIR       = os.path.join(BASE_DIR, "norm")
TOKENS_DIR     = os.path.join(BASE_DIR, "tokens")
MODEL_DIR      = "models"
JOBLIB_PATH    = os.path.join(MODEL_DIR, "stochastic_normalizer.joblib")
VQ_MODEL_PATH  = os.path.join(MODEL_DIR, "ohlcv_vqvae_encoder.pth")

# Time range (UTC)
START        = datetime.datetime(2018, 4, 17, tzinfo=datetime.timezone.utc)
END          = datetime.datetime.now(datetime.timezone.utc)
START_TS     = int(START.timestamp())
END_TS       = int(END.timestamp())
LIMIT        = 2000  # max per request

# Normalization
WINDOW_SIZE  = 100
EPS          = 1e-9

# VQ-VAE
K_CODES       = 2048
LATENT_D      = 128
BETA          = 0.25
DECAY_GAMMA   = 0.99
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS  = ["Open", "High", "Low", "Close", "Volume"]
# ────────────────────────────────────────────────────────────────────────

# ensure directories exist
for d in (MARKET_DIR, NORM_DIR, TOKENS_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

class RowStochasticNormalizer:
    """Stochastic min-max normaliser with a fixed backward-looking window."""
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

def purge(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows before the first valid row (no NaNs & 1d continuity)."""
    df = df.reset_index(drop=True)
    while True:
        if df.empty:
            return df
        mask = df[FEATURE_COLS].isnull().any(axis=1)
        if mask.any():
            df = df.iloc[mask.idxmax()+1:].reset_index(drop=True)
            continue
        diffs = df["Time"].diff().iloc[1:]
        bad   = diffs[diffs != pd.Timedelta(days=1)]
        if not bad.empty:
            df = df.iloc[bad.index[0]:].reset_index(drop=True)
            continue
        break
    return df

class Encoder(nn.Module):
    """Must match training encoder."""
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, d_latent)
        )
    def forward(self, x): return self.net(x)

def fetch_ohlcv() -> pd.DataFrame:
    """Fetch full OHLCV history for WAVES/USDT daily."""
    to_ts    = END_TS
    all_data = []
    pbar     = tqdm(desc=f"{FSYM}/{TSYM}", unit="batch")
    while True:
        resp  = requests.get(API_URL, params={
            "fsym": FSYM, "tsym": TSYM, "limit": LIMIT, "toTs": to_ts
        })
        batch = resp.json().get("Data", {}).get("Data", [])
        if not batch:
            pbar.update(1)
            break
        for e in batch:
            ts = e.get("time")
            if ts is None or ts < START_TS:
                continue
            all_data.append({
                "Time":   datetime.datetime.fromtimestamp(ts, datetime.timezone.utc),
                "Open":   float(e["open"]),
                "High":   float(e["high"]),
                "Low":    float(e["low"]),
                "Close":  float(e["close"]),
                "Volume": float(e["volumefrom"])
            })
        earliest = batch[0]["time"]
        if earliest <= START_TS:
            pbar.update(1)
            break
        to_ts = earliest - 1
        pbar.update(1)
        time.sleep(0.25)
    df = pd.DataFrame(all_data)
    return (
        df.drop_duplicates("Time")
          .sort_values("Time")
          .reset_index(drop=True)
          .astype({c: "float32" for c in FEATURE_COLS})
    )

def main():
    # 1. Fetch & sanitize
    df_raw   = fetch_ohlcv()
    if df_raw.empty:
        print("No data fetched.")
        return
    df_clean = purge(df_raw)

    # 2. Save market data
    df_clean_out = df_clean.copy()
    df_clean_out["Time"] = df_clean_out["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    market_path = os.path.join(MARKET_DIR, f"{FSYM}{TSYM}_{TIMEFRAME}.csv")
    df_clean_out.to_csv(market_path, index=False, float_format="%.6f")
    print(f"✔ Market data saved: {market_path}")

    # 3. Normalize
    if not os.path.exists(JOBLIB_PATH):
        raise FileNotFoundError(f"Missing normalizer: {JOBLIB_PATH}")
    normalizer = joblib.load(JOBLIB_PATH)

    df_norm     = normalizer.transform(df_clean)
    df_norm_out = df_norm.copy()
    df_norm_out["Time"] = df_norm_out["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    norm_path   = os.path.join(NORM_DIR, f"NORM_{FSYM}{TSYM}_{TIMEFRAME}.csv")
    df_norm_out.to_csv(norm_path, index=False, float_format="%.6f")
    print(f"✔ Normalized data saved: {norm_path}")

    # 4. Tokenize
    if not os.path.exists(VQ_MODEL_PATH):
        raise FileNotFoundError(f"Missing VQ-VAE model: {VQ_MODEL_PATH}")
    chkpt = torch.load(VQ_MODEL_PATH, map_location=DEVICE)
    enc   = Encoder(in_dim=5, d_latent=LATENT_D).to(DEVICE)
    vq    = VectorQuantize(
        dim=LATENT_D, codebook_size=K_CODES,
        decay=DECAY_GAMMA, kmeans_init=False,
        commitment_weight=BETA
    ).to(DEVICE)
    enc.load_state_dict(chkpt["encoder_state"])
    vq.load_state_dict(chkpt["vq_state"])
    enc.eval(); vq.eval()

    tensors = torch.tensor(
        df_norm[FEATURE_COLS].values,
        dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        out   = vq(enc(tensors))
        codes = out[1] if isinstance(out, tuple) else vq(enc(tensors))[1]
    tokens = codes.cpu().numpy()

    df_tok = pd.DataFrame({
        "Time":  df_norm_out["Time"],
        "Token": tokens
    })
    tok_path = os.path.join(TOKENS_DIR, f"TOKN_{FSYM}{TSYM}_{TIMEFRAME}.csv")
    df_tok.to_csv(tok_path, index=False)
    print(f"✔ Tokens saved: {tok_path}")

if __name__ == "__main__":
    main()
