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
from scipy.integrate import solve_ivp

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
# GENERIC ODE PROJECTION FUNCTIONS
# ============================================================================

def flow_matching_projection_differentiable(x: torch.Tensor, flow_model: FlowVelocityNet, T: float = 2.0, num_steps: int = 10) -> torch.Tensor:
    """
    Differentiable ODE projection using PyTorch operations.
    
    This function integrates the ODE backward in time from T to 0 using Euler's method.
    It preserves gradients and can be used during training.
    
    The ODE being solved is: dx/dt = -v(x, t)
    We integrate backward from t=T to t=0 to project onto the manifold.
    
    Arguments
    ---------
    x : torch.Tensor
        Input tensor of shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]
    flow_model : FlowVelocityNet
        Trained velocity network (will be trained in another file)
    T : float, default=2.0
        Time horizon for integration
    num_steps : int, default=40
        Number of integration steps
        
    Returns
    -------
    projected_x : torch.Tensor
        Projected points on the learned manifold, same shape as input
    """
    original_shape = x.shape
    
    # Handle sequence dimension
    if x.dim() == 3:
        batch_size, seq_len, input_dim = x.shape
        x = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        squeeze_output = True
    else:
        squeeze_output = False
        input_dim = x.shape[-1]
    
    batch_size_flat = x.shape[0]
    
    # Simple Euler integration for ODE (differentiable)
    dt = T / num_steps
    x_current = x.clone()
    
    for step in range(num_steps):
        t = T - step * dt  # Time goes from T to 0
        
        # Create input for flow matching model: [features, time]
        t_tensor = torch.full(
            (batch_size_flat, 1), t, dtype=x.dtype, device=x.device
        )
        inp = torch.cat([x_current, t_tensor], dim=1)  # [N, input_dim + 1]
        
        # Get velocity from flow matching model
        # Note: We use no_grad here because the flow_model should be pre-trained
        # If you want to train it jointly, remove the no_grad()
        v = flow_model(inp)  # [N, input_dim]
        
        # Euler step: x = x - v * dt (negative for time-reversed flow)
        x_current = x_current - v * dt
    
    # Reshape back to original shape
    if squeeze_output:
        x_current = x_current.view(original_shape)
    
    return x_current

def flow_matching_projection_scipy(
    points: np.ndarray,
    flow_model: FlowVelocityNet,
    T: float = 2.0,
    num_steps: int = 40,
    rtol: float = 1e-6,
    atol: float = 1e-8
) -> np.ndarray:
    """
    Non-differentiable ODE projection using scipy's solve_ivp for inference.
    
    This function uses a more accurate ODE solver (RK45) for inference.
    It does not preserve gradients and should only be used during evaluation.
    
    Arguments
    ---------
    points : np.ndarray
        Input points of shape (n, input_dim) or (input_dim,)
    flow_model : FlowVelocityNet
        Trained velocity network
    T : float, default=2.0
        Time horizon for integration
    num_steps : int, default=40
        Number of evaluation points for the ODE solver
    rtol : float, default=1e-6
        Relative tolerance for ODE solver
    atol : float, default=1e-8
        Absolute tolerance for ODE solver
        
    Returns
    -------
    projected_points : np.ndarray
        Projected points on the learned manifold, same shape as input
    """
    # Get device from flow_model
    device = next(flow_model.parameters()).device
    input_dim = flow_model.input_dim
    
    # Ensure points is 2D
    if points.ndim == 1:
        points = points.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    n_points = points.shape[0]
    projected_points = []
    
    for i in range(n_points):
        point = points[i]
        
        def ode_rhs(t, x_flat):
            """Right-hand side of the ODE: dx/dt = -v(x, t)"""
            x = torch.from_numpy(x_flat).float().unsqueeze(0).to(device)
            t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)
            inp = torch.cat([x, t_tensor], dim=1)  # [1, input_dim + 1]
            with torch.no_grad():
                v = flow_model(inp).cpu().numpy().squeeze()
            return v  # Negative for time-reversed flow
        
        # Integrate ODE backward in time from T to 0
        t_eval = np.linspace(T, 0, num_steps)
        sol = solve_ivp(
            ode_rhs, [T, 0], point, t_eval=t_eval,
            method='RK45', rtol=rtol, atol=atol
        )
        
        # Take the final point (at t=0)
        projected_point = sol.y[:, -1]
        projected_points.append(projected_point)
    
    projected_points = np.stack(projected_points, axis=0)
    
    if single_point:
        projected_points = projected_points.squeeze(0)
    
    return projected_points

def flow_matching_projection(
    x: torch.Tensor,
    flow_model: FlowVelocityNet,
    T: float = 2.0,
    num_steps: int = 40,
    differentiable: bool = True
) -> torch.Tensor:
    """
    Generic flow matching projection function.
    
    This is a convenience wrapper that chooses between differentiable and
    non-differentiable projection based on the differentiable flag.
    
    Arguments
    ---------
    x : torch.Tensor
        Input tensor of shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]
    flow_model : FlowVelocityNet
        Trained velocity network
    T : float, default=2.0
        Time horizon for integration
    num_steps : int, default=40
        Number of integration steps
    differentiable : bool, default=True
        If True, uses differentiable PyTorch-based projection (for training).
        If False, uses scipy-based projection (for inference, more accurate).
        
    Returns
    -------
    projected_x : torch.Tensor
        Projected points on the learned manifold, same shape as input
    """
    if differentiable:
        return flow_matching_projection_differentiable(x, flow_model, T, num_steps)
    else:
        # Convert to numpy, apply scipy projection, convert back
        original_shape = x.shape
        x_np = x.detach().cpu().numpy()
        
        # Handle sequence dimension
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            x_np = x_np.reshape(-1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        projected_np = flow_matching_projection_scipy(x_np, flow_model, T, num_steps)
        projected = torch.from_numpy(projected_np).float().to(x.device)
        
        if squeeze_output:
            projected = projected.view(original_shape)
        
        return projected

# ============================================================================
# FLOW MATCHING DATA PREPARATION (VELOCITY ADVECTION)
# ============================================================================

def prepare_flow_matching_data(
    train_init: np.ndarray,
    T: float = 2.0,
    num_timesteps: int = 30,
    velocity_scale: float = 0.5,
    velocity_cov_scale: float = 1.0,
    seed: Optional[int] = None
) -> dict:
    """
    Prepare flow matching data by sampling velocities and creating trajectories.
    
    This function implements the velocity advection process:
    1. Sample velocities from a Gaussian distribution
    2. Create trajectories Y_n^t = X_n + v_n * t for t in [0, T]
    3. Prepare training data for the velocity network
    
    Arguments
    ---------
    train_init : np.ndarray
        Initial training points, shape [N, input_dim]
    T : float, default=2.0
        Time horizon for trajectories
    num_timesteps : int, default=30
        Number of time steps in the trajectory
    velocity_scale : float, default=0.5
        Scale factor for velocity magnitudes after normalization
    velocity_cov_scale : float, default=1.0
        Scale factor for velocity covariance matrix
    seed : int or None, default=None
        Random seed for reproducibility
        
    Returns
    -------
    flow_data : dict
        Dictionary containing:
        - 'velocities': np.ndarray, shape [N, input_dim] - sampled velocities
        - 'Ynt': np.ndarray, shape [N, num_timesteps, input_dim] - trajectories
        - 't_vals': np.ndarray, shape [num_timesteps] - time values
        - 'T': float - time horizon
        - 'num_timesteps': int - number of time steps
        - 'input_flat': np.ndarray, shape [N * num_timesteps, input_dim + 1] - training inputs
        - 'v_targets': np.ndarray, shape [N * num_timesteps, input_dim] - training targets
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = train_init.shape[0]
    input_dim = train_init.shape[1]
    
    # Sample velocities from Gaussian
    mean = np.zeros(input_dim)
    cov = velocity_cov_scale * np.eye(input_dim)
    velocities = np.random.multivariate_normal(mean, cov, size=N)
    
    # Normalize velocities and scale
    velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / (velocity_norms + 1e-8) * velocity_scale
    
    # Create time grid
    t_vals = np.linspace(0, T, num_timesteps)
    
    # Compute trajectories: Y_n^t = X_n + v_n * t
    Xn = train_init[:, None, :]  # [N, 1, input_dim]
    vn = velocities[:, None, :]   # [N, 1, input_dim]
    t_grid = t_vals[None, :, None]  # [1, num_timesteps, 1]
    Ynt = Xn + vn * t_grid  # [N, num_timesteps, input_dim]
    
    # Prepare training data for velocity network
    # Input: [Y_n^t, t], Target: v_n
    Ynt_flat = Ynt.reshape(-1, input_dim)  # [N * num_timesteps, input_dim]
    t_flat = np.tile(t_vals, N)[:, None]  # [N * num_timesteps, 1]
    input_flat = np.concatenate([Ynt_flat, t_flat], axis=1)  # [N * num_timesteps, input_dim + 1]
    v_targets = np.repeat(velocities, num_timesteps, axis=0)  # [N * num_timesteps, input_dim]
    
    return {
        'velocities': velocities,
        'Ynt': Ynt,
        't_vals': t_vals,
        'T': T,
        'num_timesteps': num_timesteps,
        'input_flat': input_flat,
        'v_targets': v_targets
    }

