"""
Training function for constrained models (probabilistic, regular, projected).

This module provides a training function for:
1. Probabilistic models
2. Regular transformer models
3. Projected transformer models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Callable
import numpy as np

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Model architecture parameters
D_MODEL: int = 512
NHEAD: int = 8
D_HID: int = 2048
NLAYERS: int = 6
DROPOUT: float = 0.5
DT: float = 1.0

# Probabilistic-specific parameters
PROB_HIDDEN_DIM: int = 128
PROB_NUM_LAYERS: int = 4
PROB_NUM_PARTICLES: int = 100

# Projected-specific parameters
USE_INTERNAL_PROJECTION: bool = True

# Training parameters
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 500
BATCH_SIZE: int = 32
TRAIN_VAL_SPLIT: float = 0.8

# Optimization parameters
GRADIENT_CLIP_NORM: float = 1.0
SCHEDULER_PATIENCE: int = 15
SCHEDULER_FACTOR: float = 0.5
EARLY_STOPPING_PATIENCE: int = 30

# Other parameters
SEED: int = None
VERBOSE: bool = True


import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    lr: float,
    num_epochs: int,
    device,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    scheduler_patience: int = 300,
    scheduler_factor: float = 0.8,
    early_stop: int = 30,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    train_losses, val_losses, lrs = [], [], []
    best_val, best_epoch = float("inf"), -1
    best_state, patience = None, 0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        tr = 0.0
        n_tr = 0
        for batch in train_loader:
            # assume (x, y) batches
            x, y = batch
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            bs = x.shape[0]
            tr += loss.item() * bs
            n_tr += bs

        tr /= n_tr
        train_losses.append(tr)

        # ---- val ----
        model.eval()
        va = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                loss = F.mse_loss(model(x), y)

                bs = x.shape[0]
                va += loss.item() * bs
                n_va += bs

        va /= n_va
        val_losses.append(va)

        # ---- scheduler ----
        sched.step(va)
        lrs.append(opt.param_groups[0]["lr"])

        # ---- best model ----
        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose and ((epoch == 0) or ((epoch + 1) % 20 == 0)):
            print(
                f"epoch {epoch+1:4d} | train {tr:.3e} | val {va:.3e} | lr {opt.param_groups[0]['lr']:.1e}"
            )

        if early_stop and patience >= early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logs = dict(
        train_losses=train_losses,
        val_losses=val_losses,
        lrs=lrs,
        best_val_loss=best_val,
        best_epoch=best_epoch,
    )
    return model, logs

