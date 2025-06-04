#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TokenWindowDataset(Dataset):
    """
    Slides a context window of fixed size across each CSV in data/tokens/,
    returning (context[100], next_token) pairs with stride=1.
    """
    def __init__(self, csv_dir: str, context_size: int = 100):
        self.context_size = context_size
        self.file_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        self.tokens_list = []
        self.window_map  = []

        # Load each file and record all valid window starts
        for file_idx, path in enumerate(self.file_paths):
            df = pd.read_csv(path, usecols=["Token"])
            tokens = df["Token"].to_numpy(dtype=np.int64)
            self.tokens_list.append(tokens)
            n = len(tokens)
            for start in range(n - context_size):
                self.window_map.append((file_idx, start))

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        file_idx, start = self.window_map[idx]
        tokens = self.tokens_list[file_idx]
        context = tokens[start : start + self.context_size]
        label   = tokens[start + self.context_size]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(label,   dtype=torch.long),
        )

class NextTokenCollator:
    """
    Batches context windows and next-token labels.
    """
    def __call__(self, batch):
        contexts, labels = zip(*batch)
        contexts = torch.stack(contexts, dim=0)
        labels   = torch.stack(labels,   dim=0)
        return {"tokens": contexts, "labels": labels}
