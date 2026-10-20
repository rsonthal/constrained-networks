"""
Flow Matching Projection Functions

This module provides generic flow matching projection functions that work with
any input dimension. The projection learns the manifold structure from data
using a velocity network that will be trained in another file.

Flow matching works by:
1. Learning a velocity field v(x, t) that maps points to velocities
2. Using ODE integration to project points back onto the learned manifold
3. The manifold structure is learned from the training data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ============================================================================
# GENERIC VELOCITY NETWORK
# ============================================================================

class FlowVelocityNet(nn.Module):
    """
    Generic velocity network for flow matching that works with any input dimension.
    
    The velocity network takes (x, t) as input and outputs a velocity vector v(x, t).
    This network will be trained in another file to learn the manifold structure
    from the data.
    
    Parameters
    ----------
    input_dim : int
        Dimension of the input features (e.g., 3 for sphere, 9 for SO(3), 16 for SE(3)).
        This is the dimension of the manifold.
    hidden_dim : int, default=256
        Hidden dimension for the network layers.
    num_layers : int, default=8
        Number of layers in the network.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        layers = []
        
        # Input: features + time = input_dim + 1
        input_size = input_dim + 1

        layers.extend([nn.Linear(input_size, hidden_dim),
                       nn.LayerNorm(hidden_dim),
                       nn.GELU(),
                       nn.Dropout(dropout)])
        
        # Build network with LayerNorm, GELU, and Dropout
        for i in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim),
                           nn.LayerNorm(hidden_dim),
                           nn.GELU(),
                           nn.Dropout(dropout)])
        
        # Output velocity (same dimension as input)
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the velocity network.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim + 1] where the last
            dimension is the time t, and the first input_dim dimensions are
            the point coordinates.
            
        Returns
        -------
        v : torch.Tensor
            Velocity vector of shape [batch_size, input_dim]
        """
        v = self.net(x)
        return v

# ============================================================================
# FLOW MATCHING DATA PREPARATION (VELOCITY ADVECTION)
# ============================================================================
def prepare_flow_matching_data(
    train_init: np.ndarray,
    num_timesteps: int = 30,
    eps: float = 1e-2,
    seed: Optional[int] = None,
) -> dict:

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    N, input_dim = train_init.shape

    # Gaussian noise: one endpoint for each data point
    noise = rng.standard_normal(size=(N, input_dim))

    # Avoid t = 0 because beta_dot = 1/(2 sqrt(t))
    t_vals = np.linspace(eps, 1.0, num_timesteps)

    # Shapes for broadcasting
    X_data = train_init[:, None, :]       # [N, 1, d]
    X_noise = noise[:, None, :]           # [N, 1, d]
    t = t_vals[None, :, None]             # [1, T, 1]

    # s(t) and alpha(t)
    s = 35*t**4 - 84*t**5 + 70*t**6 - 20*t**7
    alpha = 1.0 - s
    beta = np.sqrt(t)

    # derivatives
    alpha_dot = -140*t**3 * (1.0 - t)**3
    beta_dot = 0.5 / np.sqrt(t)

    # Interpolated states
    X_t = alpha * X_data + beta * X_noise

    # Conditional flow-matching velocity targets
    V_t = alpha_dot * X_data + beta_dot * X_noise

    # Flatten over data points and times
    X_flat = X_t.reshape(-1, input_dim)
    V_flat = V_t.reshape(-1, input_dim)
    t_flat = np.tile(t_vals, N)[:, None]

    input_flat = np.concatenate([X_flat, t_flat], axis=1)

    return {
        "noise": noise,
        "X_t": X_t,
        "t_vals": t_vals,
        "input_flat": input_flat,
        "v_targets": V_flat,
        "num_timesteps": num_timesteps,
    }