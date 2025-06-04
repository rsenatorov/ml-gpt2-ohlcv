#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import torch
import torch.nn as nn
from network.blocks.positional_encoding import PositionalEncoding
from network.blocks.transformer_block   import TransformerBlock

class GPT2TimeSeries(nn.Module):
    """
    GPT-2 Small variant for time-series next-token prediction.
    """
    def __init__(
        self,
        vocab_size:   int = 2048,
        context_size: int = 100,
        d_model:      int = 768,
        n_heads:      int = 12,
        d_ff:         int = 3072,
        n_layers:     int = 12,
        dropout:      float = 0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(context_size, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f      = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        self.context_size = context_size

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: [B, L] with L <= context_size
        returns logits for next token: [B, vocab_size]
        """
        bsz, seq_len = tokens.size()
        assert seq_len <= self.context_size

        x = self.token_emb(tokens)        # [B, L, D]
        x = self.pos_enc(x)               # +pos
        x = self.drop(x)

        # build causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        attn_mask = ~mask

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        last = x[:, -1, :]                # [B, D]
        logits = self.lm_head(last)       # [B, vocab_size]
        return logits
