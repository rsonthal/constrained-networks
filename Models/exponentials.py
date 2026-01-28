"""
Exponential Map Projections for Sphere, SO(3), and SE(3)

This module provides exponential map projection functions for three different geometries:
1. Sphere (S²): Exponential map on the unit sphere
2. SO(3): Exponential map for rotation matrices
3. SE(3): Exponential map for rigid body transformations
"""

import torch
import torch.nn.functional as F
from typing import Optional

# ============================================================================
# SPHERE EXPONENTIAL MAP
# ============================================================================

def sphere(x_base: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    x_next = exp_{x_base}(dt * v), with v = proj_{T_{x_base}S}(u).

    Inputs:
      x_base: (B, 3)  (assumed near/on the sphere; we renormalize)
      u:      (B, 3)  (ambient velocity; we project to tangent)
      dt:     scalar tensor (e.g. nn.Parameter(())) so gradients flow into dt
    """
    x_base = F.normalize(x_base, p=2, dim=-1)

    # tangent projection
    v = u - (u * x_base).sum(dim=-1, keepdim=True) * x_base

    # scale by dt (dt is a tensor/Parameter -> backprop works)
    v = dt * v

    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    eps = 1e-7

    y = torch.cos(norm_v) * x_base + torch.sin(norm_v) * (v / torch.clamp(norm_v, min=eps))
    out = torch.where(norm_v <= eps, x_base, y)

    return F.normalize(out, p=2, dim=-1)
    
# ============================================================================
# Lie Group EXPONENTIAL MAP
# ============================================================================

def so3_basis(device=None, dtype = None):
    E1 = torch.tensor([[0., 0., 0.],
                       [0., 0., -1.],
                       [0., 1.,  0.]])
    E2 = torch.tensor([[0., 0.,  1.],
                       [0., 0.,  0.],
                       [-1.,0.,  0.]])
    E3 = torch.tensor([[0., -1., 0.],
                       [1.,  0., 0.],
                       [0.,  0., 0.]])
    Ei = torch.stack([E1, E2, E3], dim=0)
    if device is not None: Ei = Ei.to(device)
    return Ei  # (3,3,3)

# rotation generators embedded in 4x4
def rot_hom(R,device=None, dtype=None):
    E = torch.zeros(4, 4, device=device, dtype=dtype)
    E[:3, :3] = R
    return E
    
def trans_hom(axis,device=None, dtype=None):
    E = torch.zeros(4, 4, device=device, dtype=dtype)
    E[axis, 3] = 1.0
    return E

def se3_basis(device=None, dtype=None):
    
    R1 = torch.tensor([[0., 0., 0.],
                       [0., 0., -1.],
                       [0., 1.,  0.]], device=device, dtype=dtype)
    R2 = torch.tensor([[0., 0.,  1.],
                       [0., 0.,  0.],
                       [-1.,0.,  0.]], device=device, dtype=dtype)
    R3 = torch.tensor([[0., -1., 0.],
                       [1.,  0., 0.],
                       [0.,  0., 0.]], device=device, dtype=dtype)

    Ei = torch.stack([
        rot_hom(R1, device=device, dtype=dtype), rot_hom(R2, device=device, dtype=dtype), rot_hom(R3, device=device, dtype=dtype),  # Rx,Ry,Rz
        trans_hom(0, device=device, dtype=dtype), trans_hom(1, device=device, dtype=dtype), trans_hom(2, device=device, dtype=dtype) # Tx,Ty,Tz
    ], dim=0)
    return Ei  # (6,4,4)

def so3(g9: torch.Tensor, a: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Left-invariant step on SO(3) using matrix exponential:
        g_next = exp(dt * sum_i a_i E_i) @ g

    Supports:
      g9: (B,9)    and a: (B,3)
      g9: (B,S,9)  and a: (B,S,3)

    dt: scalar tensor (can be nn.Parameter), gradients flow through it.
    """
    if g9.dim() == 2:
        B, _ = g9.shape
        G  = g9.reshape(B, 3, 3).contiguous()
        aa = a.reshape(B, 3).contiguous()

        Ei = so3_basis(device=G.device, dtype=G.dtype)            # (3,3,3)
        Xi = (aa[:, :, None, None] * Ei[None]).sum(dim=1)         # (B,3,3)
        G_next = torch.matrix_exp(dt * Xi) @ G                    # (B,3,3)
        return G_next.view(B, 9)

    if g9.dim() == 3:
        B, S, _ = g9.shape
        G  = g9.reshape(B*S, 3, 3).contiguous()
        aa = a.reshape(B*S, 3).contiguous()

        Ei = so3_basis(device=G.device, dtype=G.dtype)            # (3,3,3)
        Xi = (aa[:, :, None, None] * Ei[None]).sum(dim=1)         # (BS,3,3)
        G_next = torch.matrix_exp(dt * Xi) @ G                    # (BS,3,3)
        return G_next.view(B, S, 9)

    raise ValueError("g9 must have shape (B,9) or (B,S,9)")

def se3(G16: torch.Tensor, a: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    G16: (B,16) flattened SE(3) in homogeneous coordinates
    a  : (B,6)
    dt : scalar tensor (e.g. nn.Parameter)
    """
    B, _ = G16.shape
    G = G16.view(B, 4, 4)

    Ei = se3_basis(device=G.device, dtype=G.dtype)      # (6,4,4)
    Xi = (a[:, :, None, None] * Ei[None]).sum(dim=1)    # (B,4,4)

    G_next = torch.matrix_exp(dt * Xi) @ G              # (B,4,4)
    return G_next.view(B, 16)
    