#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Robert Senatorov
# All rights reserved.

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_steps`, then cosine anneal over `T_max` epochs.
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.T_max        = T_max
        self.eta_min      = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [lr * factor for lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.T_max)
        return [
            self.eta_min + (lr - self.eta_min) *
            0.5 * (1 + math.cos(math.pi * progress))
            for lr in self.base_lrs
        ]
