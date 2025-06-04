#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer decoder block with:
      - causal self-attention
      - GELU feed-forward
      - residual connections
    """
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
                        d_model, n_heads,
                        dropout=dropout,
                        batch_first=True
                    )
        self.drop = nn.Dropout(dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                        nn.Dropout(dropout),
                    )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        # Self-attention
        res = x
        x   = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x   = res + self.drop(x)

        # Feed-forward
        res = x
        x   = self.ln2(x)
        x   = res + self.ff(x)
        return x
