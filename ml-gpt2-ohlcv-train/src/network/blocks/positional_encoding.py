#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Learned positional embeddings for a fixed context length.
    """
    def __init__(self, context_size: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, context_size, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pos_emb[:, : x.size(1), :]
