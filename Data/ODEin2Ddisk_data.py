"""
ODEin2Ddisk_data.py

Analytic (grid-free) projected dynamical system on the closed unit disk,
refactored into a "sphere_data-like" API:

    - create_training_data(...) -> train_init, train_final, test_init, test_final

This module preserves the dynamics from the original ODEin2Ddisk.py:
    F(x) = J x + alpha x
with a projection of the velocity onto the tangent cone at the disk boundary.

Notes
-----
- The original script used explicit Euler and only re-projected the STATE
  to the disk at the *start* of each step (not after the step). By default,
  we keep that behavior for data fidelity. If you want strict invariance
  (always inside the disk), set `project_after_step=True`.

- `renormalize_after_noise` matches the sphere_data flag name but here means:
  "project final points back into the disk after noise".
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# -----------------------------
# Core field (matches original)
# -----------------------------

J_DEFAULT = np.array([[0.0, -1.0],
                      [1.0,  0.0]], dtype=float)


from typing import NamedTuple


class DiskODEParams(NamedTuple):
    """Parameters for the analytic disk ODE."""
    alpha: float = 0.5                # outward radial strength (matches original)
    rotation_strength: float = 1.0    # multiplies Jx (1.0 matches original)
    radius: float = 1.0               # disk radius (original used 1.0)
    boundary_tol: float = 1e-6        # tolerance for "near boundary"



def F(x: np.ndarray, params: DiskODEParams = DiskODEParams()) -> np.ndarray:
    """
    Unconstrained analytic vector field:
        F(x) = rotation_strength * (J x) + alpha * x.

    Parameters
    ----------
    x : (..., 2)
        Point(s) in R^2.
    params : DiskODEParams
        ODE parameters.

    Returns
    -------
    v : (..., 2)
        Velocity at x (same batch shape as x).
    """
    x = np.asarray(x, dtype=float)
    # batch multiply: for each row x_i, compute J @ x_i  == x_i @ J^T
    rot = x @ (J_DEFAULT.T)
    return params.rotation_strength * rot + params.alpha * x


def project_points_to_disk(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    Radially project point(s) onto the closed disk of given radius.

    Parameters
    ----------
    x : (..., 2)
    radius : float

    Returns
    -------
    x_proj : (..., 2)
    """
    x = np.asarray(x, dtype=float)
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    # avoid division by zero
    scale = np.where(r > radius, radius / np.maximum(r, 1e-12), 1.0)
    return x * scale


def projected_F(x: np.ndarray, params: DiskODEParams = DiskODEParams()) -> np.ndarray:
    """
    Projected vector field onto the tangent cone of the closed disk:
        K = { x : ||x|| <= radius }.

    Rule (matches original):
    - If x is strictly interior: v = F(x).
    - If x is on/near boundary: remove outward normal component of v.

    Parameters
    ----------
    x : (..., 2)
        Point(s) where we evaluate the projected velocity.
    params : DiskODEParams

    Returns
    -------
    v_proj : (..., 2)
    """
    x = np.asarray(x, dtype=float)
    r = np.linalg.norm(x, axis=-1)  # (...)
    v = F(x, params=params)         # (...,2)

    # Interior mask (strictly inside)
    interior = r < (params.radius - params.boundary_tol)
    if np.all(interior):
        return v

    # For boundary points, compute outward normal and remove outward component
    # n = x / ||x||. For x=0, leave v unchanged (though it is interior anyway).
    r_safe = np.maximum(r, 1e-12)
    n = x / r_safe[..., None]
    vn = np.sum(v * n, axis=-1)  # (...)

    # Apply correction only if near boundary AND pointing outward
    boundary = ~interior
    outward = vn > 0
    mask = boundary & outward

    if np.any(mask):
        v = v.copy()
        v[mask] = v[mask] - vn[mask, None] * n[mask]
    return v


# -----------------------------
# Simulation / advection
# -----------------------------

def advect_points_in_disk(
    init_pos: np.ndarray,
    num_steps: int = 100,
    dt: float = 0.01,
    params: DiskODEParams = DiskODEParams(),
    project_after_step: bool = False,
    return_trajectory: bool = False,
) -> np.ndarray:
    """
    Advect one or many particles under the projected field using explicit Euler,
    matching the original ODEin2Ddisk.py behavior.

    Parameters
    ----------
    init_pos : (2,) or (N, 2)
        Initial position(s).
    num_steps : int
        Number of Euler steps.
    dt : float
        Step size.
    params : DiskODEParams
        ODE parameters.
    project_after_step : bool
        If False (default): matches original script: project state to disk only
        at the *start* of each step.
        If True: additionally project after each Euler update (keeps strictly in-disk).
    return_trajectory : bool
        If True, returns (num_steps+1, 2) for a single particle.

    Returns
    -------
    final_pos : (2,) or (N, 2) or trajectory
        Final positions (or trajectory if requested).
    """
    x0 = np.asarray(init_pos, dtype=float)

    single = (x0.ndim == 1)
    if single:
        x = x0.copy()
        if return_trajectory:
            traj = np.zeros((num_steps + 1, 2), dtype=float)
            traj[0] = x
    else:
        if x0.shape[1] != 2:
            raise ValueError(f"init_pos must have shape (N,2); got {x0.shape}")
        x = x0.copy()

    for k in range(num_steps):
        # original: clamp state to disk at the start of each step
        x = project_points_to_disk(x, radius=params.radius)

        v = projected_F(x, params=params)
        x = x + dt * v

        if project_after_step:
            x = project_points_to_disk(x, radius=params.radius)

        if single and return_trajectory:
            traj[k + 1] = x

    if single and return_trajectory:
        return traj
    return x


# -----------------------------
# Sampling and dataset API
# -----------------------------

def sample_initial_conditions(
    N: int,
    radius: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample N points uniformly (in area) from the disk of given radius.
    Matches the original script's sampling (sqrt trick).

    Returns
    -------
    pts : (N,2)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(N)
    theta = 2.0 * np.pi * rng.random(N)
    r = radius * np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def create_training_data(
    num_samples: int = 5000,
    num_steps: int = 100,
    dt: float = 0.01,
    noise_level: float = 0.0,
    save_particles_path: Optional[str] = None,
    renormalize_after_noise: bool = False,
    seed: Optional[int] = None,
    params: DiskODEParams = DiskODEParams(),
    project_after_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training pairs (x0 -> xT) by advecting random disk points under the
    analytic projected vector field from the original ODEin2Ddisk.py.

    This mirrors the *shape* and *return convention* of sphere_data.create_training_data,
    but without any grid inputs.

    Pipeline:
      1) Sample `num_samples` initial points uniformly in the disk.
      2) Advect all points for `num_steps` of size `dt` via explicit Euler.
      3) Optionally add Gaussian noise to the final points.
      4) Optionally reproject final points back into the disk (after noise).
      5) Deterministic 80/20 split into train/test.

    Parameters
    ----------
    num_samples : int
        Number of (x0, xT) pairs.
    num_steps : int
        Number of Euler steps.
    dt : float
        Step size.
    noise_level : float
        Std. dev. of i.i.d. Gaussian noise added to final positions.
    save_particles_path : str or None
        If provided, saves the *initial* points to this .npy path (like sphere_data).
    renormalize_after_noise : bool
        If True, projects final points back into the disk after adding noise.
    seed : int or None
        Seed for reproducible sampling (and noise).
    params : DiskODEParams
        ODE parameters.
    project_after_step : bool
        If True, keep points in-disk after each Euler update.
        Default False to better match original script.

    Returns
    -------
    train_init, train_final, test_init, test_final : (.,2) arrays
    """
    rng = np.random.default_rng(seed)

    initial_points = sample_initial_conditions(num_samples, radius=params.radius, rng=rng)

    if save_particles_path is not None:
        # save initial points only (mirrors sphere_data behavior)
        import os
        os.makedirs(os.path.dirname(save_particles_path) or ".", exist_ok=True)
        np.save(save_particles_path, initial_points)

    final_points = advect_points_in_disk(
        initial_points,
        num_steps=num_steps,
        dt=dt,
        params=params,
        project_after_step=project_after_step,
        return_trajectory=False,
    )

    if noise_level and noise_level > 0.0:
        final_points = final_points + rng.normal(loc=0.0, scale=noise_level, size=final_points.shape)

    if renormalize_after_noise:
        final_points = project_points_to_disk(final_points, radius=params.radius)

    # deterministic 80/20 split (keep order stable for reproducibility)
    train_size = int(0.8 * num_samples)
    train_init = initial_points[:train_size]
    train_final = final_points[:train_size]
    test_init = initial_points[train_size:]
    test_final = final_points[train_size:]

    return train_init, train_final, test_init, test_final
