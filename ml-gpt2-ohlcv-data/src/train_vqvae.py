#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

"""
train_vqvae.py - VQ-VAE encoder+decoder (2048-token code-book) for OHLCV candles

Now compatible with normalize_market_data.py outputs in data/norm/*.csv:
  • Reads every NORM_*.csv, uses each [Open,High,Low,Close,Volume] row as a sample
  • Continues training until ≥N_EPOCHS *and* all K_CODES have been used at least once
  • Shows high-precision loss and code-usage in the progress bar
  • Saves:
      • Encoder + codebook → models/ohlcv_vqvae_encoder.pth
      • Decoder           → models/ohlcv_vqvae_decoder.pth
      • Config JSON       → models/ohlcv_vqvae_cfg.json
      • vocab.json        → mapping each token → avg [O,H,L,C,V]
"""

import os
import glob
import math
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# ─── configuration ───────────────────────────────────────────
RAW_DIR     = os.path.join("data", "norm")
FILE_GLOB   = os.path.join(RAW_DIR, "NORM_*.csv")
MODELS_DIR  = "models"
ENC_PATH    = os.path.join(MODELS_DIR, "ohlcv_vqvae_encoder.pth")
DEC_PATH    = os.path.join(MODELS_DIR, "ohlcv_vqvae_decoder.pth")
CFG_PATH    = os.path.join(MODELS_DIR, "ohlcv_vqvae_cfg.json")
VOCAB_PATH  = os.path.join(MODELS_DIR, "vocab.json")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_ROWS  = 50_000
BATCH_SZ    = 4_096
N_EPOCHS    = 25
K_CODES     = 2048
LATENT_D    = 128
BETA        = 0.25
DECAY_GAMMA = 0.99
LR          = 3e-4
CLIP_NORM   = 1.0

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ─── model parts ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, d_latent)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, d_latent: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 256), nn.GELU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, z):
        return self.net(z)

# ─── data helpers ────────────────────────────────────────────
def count_csv_rows(paths):
    """Total data points (rows) across all normalized CSVs."""
    total = 0
    for p in paths:
        # subtract header
        with open(p, 'r') as f:
            total += sum(1 for _ in f) - 1
    return total

def infinite_csv_loader(paths, chunk_rows, batch_size, shuffle=True):
    """
    Infinite generator yielding batches of shape [batch_size, 5] 
    from all normalized CSVs in sequence.
    """
    while True:
        for path in paths:
            for chunk in pd.read_csv(path, usecols=FEATURE_COLS, chunksize=chunk_rows):
                if shuffle:
                    chunk = chunk.sample(frac=1).reset_index(drop=True)
                num = len(chunk)
                for start in range(0, num, batch_size):
                    end = min(start + batch_size, num)
                    batch = torch.tensor(
                        chunk.iloc[start:end].values,
                        dtype=torch.float32,
                    )
                    yield batch

# ─── training & saving ───────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # collect all normalized CSVs
    files = sorted(glob.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No normalized CSVs found in {RAW_DIR}")

    # instantiate models
    enc = Encoder(in_dim=5, d_latent=LATENT_D).to(DEVICE)
    dec = Decoder(d_latent=LATENT_D, out_dim=5).to(DEVICE)
    vq  = VectorQuantize(
        dim=LATENT_D,
        codebook_size=K_CODES,
        decay=DECAY_GAMMA,
        kmeans_init=True,
        kmeans_iters=10,
        commitment_weight=BETA
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters()),
        lr=LR, betas=(0.9, 0.95), weight_decay=1e-4
    )
    sched = CosineAnnealingLR(optim, T_max=N_EPOCHS)

    # compute steps per epoch
    total_rows   = count_csv_rows(files)
    steps_per_ep = math.ceil(total_rows / BATCH_SZ)
    print(f"Total rows: {total_rows}, Steps per epoch: {steps_per_ep}")

    loader = infinite_csv_loader(files, CHUNK_ROWS, BATCH_SZ, shuffle=True)

    used_codes = set()
    epoch = 1
    while True:
        enc.train(); dec.train(); vq.train()
        used_codes.clear()
        running_loss = 0.0

        pbar = tqdm(range(steps_per_ep),
                    desc=f"Epoch {epoch}/{N_EPOCHS}",
                    dynamic_ncols=True)
        for _ in pbar:
            batch = next(loader).to(DEVICE)
            optim.zero_grad()

            z_e = enc(batch)
            z_q, indices, com_loss = vq(z_e)
            used_codes.update(indices.cpu().numpy().tolist())

            loss_recon = nn.functional.mse_loss(dec(z_q), batch)
            loss = loss_recon + com_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters()),
                CLIP_NORM
            )
            optim.step()
            running_loss += loss.item()

            pbar.set_postfix(
                loss   = f"{loss.item():.6f}",
                recon  = f"{loss_recon.item():.6f}",
                commit = f"{com_loss.item():.6f}",
                codes  = f"{len(used_codes)}/{K_CODES}"
            )
        sched.step()

        mean_loss = running_loss / steps_per_ep
        print(f"Epoch {epoch}: mean loss {mean_loss:.6f}, used codes {len(used_codes)}/{K_CODES}")

        # stop when enough epochs and all codes seen
        if epoch >= N_EPOCHS and len(used_codes) == K_CODES:
            break
        epoch += 1

    # save encoder + codebook
    torch.save({
        "encoder_state": enc.state_dict(),
        "vq_state":      vq.state_dict()
    }, ENC_PATH)
    print(f"✓ Saved encoder → {ENC_PATH}")

    # save decoder
    torch.save(dec.state_dict(), DEC_PATH)
    print(f"✓ Saved decoder → {DEC_PATH}")

    # save config
    with open(CFG_PATH, "w") as fp:
        json.dump({
            "latent_dim":     LATENT_D,
            "codebook_size":  K_CODES,
            "beta":           BETA,
            "decay_gamma":    DECAY_GAMMA
        }, fp, indent=2)
    print(f"✓ Saved config  → {CFG_PATH}")

    # build vocab mapping
    print("Building vocab mapping (averaging original OHLCV per token)…")
    sums   = np.zeros((K_CODES, 5), dtype=np.float64)
    counts = np.zeros(K_CODES, dtype=np.int64)

    for path in files:
        for chunk in pd.read_csv(path, usecols=FEATURE_COLS, chunksize=CHUNK_ROWS):
            data = torch.tensor(chunk.values, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                _, idxs, _ = vq(enc(data))
            idxs = idxs.cpu().numpy()
            for code, feat in zip(idxs, chunk.values):
                sums[code]   += feat
                counts[code] += 1

    vocab = {}
    for code in range(K_CODES):
        if counts[code] > 0:
            avg = (sums[code] / counts[code]).tolist()
        else:
            avg = [0.0] * 5
        vocab[str(code)] = avg

    with open(VOCAB_PATH, "w") as fp:
        json.dump(vocab, fp, indent=2)
    print(f"✓ Saved vocab mapping → {VOCAB_PATH}")

if __name__ == "__main__":
    main()
