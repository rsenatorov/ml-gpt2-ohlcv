#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def save_ckpt(state: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
