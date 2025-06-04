#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
tokenize_normalized_data.py

For every normalized OHLCV file NORM_{pair_tf}.csv in data/norm/:
  - Loads the pretrained VQ-VAE encoder + codebook from models/ohlcv_vqvae_encoder.pth
  - Encodes each [Open,High,Low,Close,Volume] row into a single integer token
  - Writes out data/tokens/TOKN_{pair_tf}.csv with columns: Time,Token
"""

import os
import glob
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# ── CONFIG ────────────────────────────────────────────────────────────
INPUT_DIR   = os.path.join("data", "norm")
OUTPUT_DIR  = os.path.join("data", "tokens")
MODEL_PATH  = os.path.join("models", "ohlcv_vqvae_encoder.pth")

# these must match your training setup
K_CODES       = 2048
LATENT_D      = 128
BETA          = 0.25        # commitment weight (unused at inference)
DECAY_GAMMA   = 0.99        # codebook EMA decay (unused at inference)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS  = ["Open", "High", "Low", "Close", "Volume"]
# ────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Must match the Encoder used during training.
    """
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, d_latent)
        )
    def forward(self, x):
        return self.net(x)

def main():
    # Load checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Instantiate models
    enc = Encoder(in_dim=5, d_latent=LATENT_D).to(DEVICE)
    vq  = VectorQuantize(
        dim=LATENT_D,
        codebook_size=K_CODES,
        decay=DECAY_GAMMA,
        kmeans_init=False,
        commitment_weight=BETA
    ).to(DEVICE)

    # Restore weights
    enc.load_state_dict(checkpoint["encoder_state"])
    vq.load_state_dict(checkpoint["vq_state"])
    enc.eval(); vq.eval()

    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each normalized CSV
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "NORM_*.csv")))
    if not files:
        print(f"No normalized CSVs found in {INPUT_DIR}")
        return

    for path in files:
        df = pd.read_csv(path, parse_dates=["Time"])
        if df.empty:
            print(f"Skipping empty file {os.path.basename(path)}")
            continue

        # Prepare data tensor
        data = torch.tensor(df[FEATURE_COLS].values,
                            dtype=torch.float32,
                            device=DEVICE)

        # Quantize
        with torch.no_grad():
            z_e       = enc(data)
            _, ids, _ = vq(z_e)

        # Build output DataFrame
        tokens = ids.cpu().numpy()
        out_df = pd.DataFrame({
            "Time":  df["Time"],
            "Token": tokens
        })

        # Derive output filename: NORM_ADAUSDT_D1.csv → TOKN_ADAUSDT_D1.csv
        base = os.path.basename(path)
        name = os.path.splitext(base)[0].replace("NORM_", "TOKN_") + ".csv"
        out_path = os.path.join(OUTPUT_DIR, name)

        # Save
        out_df.to_csv(out_path, index=False)
        print(f"✓ Saved {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
