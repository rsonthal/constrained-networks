"""
Training function for flow matching velocity network.

This module provides a training function for the FlowVelocityNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
import numpy as np
import copy
from flow_matching import FlowVelocityNet, prepare_flow_matching_data

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Model parameters
HIDDEN_DIM: int = 256
NUM_LAYERS: int = 8

# Training parameters
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 500
BATCH_SIZE: int = 512
TRAIN_VAL_SPLIT: float = 0.8

# Flow matching data parameters
T: float = 2.0
NUM_TIMESTEPS: int = 30
VELOCITY_SCALE: float = 0.5
VELOCITY_COV_SCALE: float = 1.0

# Optimization parameters
GRADIENT_CLIP_NORM: float = 1.0
SCHEDULER_PATIENCE: int = 15
SCHEDULER_FACTOR: float = 0.5
EARLY_STOPPING_PATIENCE: int = 30

# Other parameters
SEED: int = None
VERBOSE: bool = True


def _make_loader(flow_data: Dict[str, np.ndarray],
                 batch_size: int,
                 shuffle: bool) -> DataLoader:
    x = torch.from_numpy(flow_data["input_flat"]).float()
    v = torch.from_numpy(flow_data["v_targets"]).float()
    return DataLoader(TensorDataset(x, v), batch_size=batch_size, shuffle=shuffle)


def train_flow_matching_model(
    model: FlowVelocityNet,
    train_init: np.ndarray,
    val_init: np.ndarray,
    device: torch.device,
    *,
    T: float = 2.0,
    num_timesteps: int = 30,
    velocity_scale: float = 0.5,
    velocity_cov_scale: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 500,
    batch_size: int = 512,
    grad_clip: float = 1.0,
    scheduler_patience: int = 15,
    scheduler_factor: float = 0.8,
    early_stop: int = 30,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[FlowVelocityNet, Dict[str, Any]]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    
    val_flow = prepare_flow_matching_data(
        train_init=val_init,
        T=T,
        num_timesteps=num_timesteps,
        velocity_scale=velocity_scale,
        velocity_cov_scale=velocity_cov_scale,
        seed=seed,
    )

    
    val_loader = _make_loader(val_flow, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), -1
    best_state, patience = None, 0

    for epoch in range(num_epochs):
        train_flow = prepare_flow_matching_data(
        train_init=train_init,
        T=T,
        num_timesteps=num_timesteps,
        velocity_scale=velocity_scale,
        velocity_cov_scale=velocity_cov_scale,
        seed=seed)
        
        train_loader = _make_loader(train_flow, batch_size=batch_size, shuffle=True)

        if epoch == 1:
            print(len(train_loader))
        
        model.train()
        tr = 0.0
        for x, v in train_loader:
            x, v = x.to(device), v.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), v)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr += loss.item()
        tr /= len(train_loader)
        train_losses.append(tr)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, v in val_loader:
                x, v = x.to(device), v.to(device)
                va += F.mse_loss(model(x), v).item()
        va /= len(val_loader)
        val_losses.append(va)

        sched.step(va)

        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose and ((epoch == 0) or ((epoch + 1) % 20 == 0)):
            print(f"epoch {epoch+1:4d} | train {tr:.3e} | val {va:.3e} | lr {opt.param_groups[0]['lr']:.1e}")

        if early_stop and patience >= early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logs = dict(
        train_flow=train_flow,
        val_flow=val_flow,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val,
        best_epoch=best_epoch,
    )
    return model, logs
