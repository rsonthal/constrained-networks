import argparse
import os
import sys, time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, Dict
from scipy.integrate import solve_ivp

# Set up device for GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_sphere_grid(grid_size=100, radius=1.0):
    """
    Generate a grid of points on a sphere for vector field visualization.
    
    Parameters
    ----------
    grid_size : int, default=100
        Number of samples per angular dimension. Produces a `grid_size × grid_size`
        mesh. Must be >= 2. Note that using equally spaced angles leads to higher
        sampling density near the poles (not equal-area).
    radius : float, default=1.0
        Sphere radius. Must be > 0.

    Returns
    -------
    X, Y, Z : np.ndarray
        Cartesian coordinates on the sphere surface, each with shape
        `(grid_size, grid_size)`. Rows index φ, columns index θ.
    Theta, Phi : np.ndarray
        Angular grids with the same shape as above. `Theta` ∈ [0, π] (0: +z pole),
        `Phi` ∈ [0, 2π] (0 along +x axis, increasing toward +y).
    """

    # --- Angular grids --------------------------------------------------------
    # θ (Theta): polar angle from +z (0) to -z (π)
    
    theta = np.linspace(0, np.pi, grid_size) # shape: (grid_size,)

    # φ (Phi): azimuth around z-axis from +x (0) through +y (π/2) to 2π
    phi = np.linspace(0, 2 * np.pi, grid_size) # shape: (grid_size,)

    # Meshgrid: rows vary over φ, columns over θ
    Theta, Phi = np.meshgrid(theta, phi) # both (grid_size, grid_size)

    # --- Spherical to Cartesian mapping --------------------------------------
    # x = r sinθ cosφ, y = r sinθ sinφ, z = r cosθ
    X = radius * np.sin(Theta) * np.cos(Phi)
    Y = radius * np.sin(Theta) * np.sin(Phi)
    Z = radius * np.cos(Theta)
    
    return X, Y, Z, Theta, Phi

def sample_random_points_on_sphere(n, dim = 3, radius=1.0):
    """
    Uniformly sample `n` points on the surface of the (d-1)-sphere in R^d via
    Gaussian draws followed by L2 normalization.

    Method
    ------
    Draw X ~ N(0, I_d) i.i.d., then set Y = radius * X / ||X||_2.
    By rotational invariance of the Gaussian, Y/ radius is uniform on S^{d-1}.

    Parameters
    ----------
    n : int
        Number of points to sample (>= 0). If 0, returns a (0, dim) tensor.
    radius : float, default=1.0
        Sphere radius (> 0).
    dim : int, default=3
        Ambient dimension d (>= 2). Returns points on S^{d-1} ⊂ R^d.

    Returns
    -------
    pts : torch.Tensor, shape (n, dim)
        Each row is a point on the sphere of radius `radius` in R^d.
    """
    # --- sample and normalize ---
    x = torch.randn((n, dim))
    norms = x.norm(dim=1, keepdim=True)
    y = (radius * x / norms)
    return y

def generate_random_continuous_vector_field_on_sphere(X, Y, Z, Theta, Phi, basis: str = "coordinate"):
    """
    Generate a smooth random tangent vector field on the unit sphere using low-order
    spherical-harmonic–like scalar fields and the spherical coordinate basis.

    Method
    ------
    We form two smooth scalar fields a(θ, φ), b(θ, φ) (low-order trigs) and combine
    the geometric basis vectors on S²:
        r(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ)
        ∂θ r = (cosθ cosφ, cosθ sinφ, -sinθ)                  (tangent)
        ∂φ r = (-sinθ sinφ,  sinθ cosφ,     0)                (tangent)
    The vector field is:
        V = a ∂θ r + b B_φ,  where B_φ = ∂φ r      if basis == "coordinate"
                               B_φ = (∂φ r)/sinθ   if basis == "orthonormal"
    (with a numerically safe 1/sinθ at the poles).

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Cartesian coordinates on the sphere of radius 1; each shape (m, n).
        Only used for optional projection and return shape; radius inferred.
    Theta, Phi : np.ndarray
        Angular grids (same shape as X): Theta ∈ [0, π], Phi ∈ [0, 2π].
    basis : {"coordinate", "orthonormal"}, default="coordinate"
        Choice of tangent basis. "orthonormal" scales the φ-direction by 1/sinθ.

    Returns
    -------
    U, V, W : np.ndarray
        Components of the tangent vector field in Cartesian coordinates; each (m, n).

    """

    # --- Smooth scalar coefficient fields a(θ, φ), b(θ, φ) --------------------
    a0, a1, a2 = np.random.uniform(-1, 1, size=3)
    b0, b1, b2 = np.random.uniform(-1, 1, size=3)

    a =   a0 * np.sin(Theta) * np.cos(Phi) + a1 * np.cos(2.0 * Theta) + a2 * np.sin(Theta) * np.sin(2.0 * Phi)
    
    b =   b0 * np.cos(Theta) * np.sin(Phi) + b1 * np.sin(2.0 * Phi) + b2 * np.sin(Theta) * np.cos(2.0 * Phi)

    # --- Coordinate basis on the sphere --------------------------------------
    # ∂θ r and ∂φ r as (m, n, 3) arrays
    r_theta = np.stack([np.cos(Theta) * np.cos(Phi),
                        np.cos(Theta) * np.sin(Phi),
                        -np.sin(Theta)],
                        axis=-1)
    r_phi = np.stack([-np.sin(Theta) * np.sin(Phi),
                      np.sin(Theta) * np.cos(Phi),
                      np.zeros_like(Phi)],
                      axis=-1)

    if basis == "coordinate":
        b_phi = r_phi
    else:
        r_phi = np.stack([np.sin(Phi),
                          np.cos(Phi),
                          np.zeros_like(Phi)],
                          axis=-1)
    
    # --- Assemble a ∂θ r + b B_φ ---------------------------------------------
    tangent_vecs = a[..., None] * r_theta + b[..., None] * b_phi
    
    # Normalize to unit length
    norm = np.linalg.norm(tangent_vecs, axis=-1, keepdims=True) + 1e-8
    tangent_vecs = tangent_vecs / norm
    
    U, V, W = tangent_vecs[..., 0], tangent_vecs[..., 1], tangent_vecs[..., 2]
    return U, V, W

def cart2sph_angles_batch(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.linalg.norm(p, axis=1)
    z = p[:, 2]
    theta = np.arccos(np.clip(z / np.maximum(R, 1e-16), -1.0, 1.0))  # [0,π]
    phi = (np.arctan2(p[:, 1], p[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)  # [0,2π)
    return theta, phi, R

def normalize_to_radius(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return p / np.linalg.norm(p, axis=1, keepdims=True) * R[:, None]

def project_to_tangent(p: np.ndarray, v: np.ndarray) -> np.ndarray: 
    nrm = np.linalg.norm(p, axis=1, keepdims=True) 
    n = p / nrm 
    return v - (np.sum(v * n, axis=1, keepdims=True)) * n

def bilinear_sample_scalar(grid: np.ndarray, theta: np.ndarray, phi: np.ndarray, dtheta: float, dphi: float, m_eff: int) -> np.ndarray:
    """
    Vectorized bilinear interpolation on uniform [0,π]×[0,2π) grid
    with periodic wrap in φ using m_eff = m-1 cells.
    """
    m, n = grid.shape  # rows=φ, cols=θ

    # --- θ index & weight (non-periodic) ---
    # Map θ to fractional column index in [0, n-1)
    j_float = np.clip(theta / dtheta, 0.0, (n - 1) - 1e-12)
    j0 = np.floor(j_float).astype(np.int64)        # 0 .. n-2
    j1 = np.minimum(j0 + 1, n - 1)                 # 1 .. n-1
    s = j_float - j0                               # [0,1)

    # --- φ index & weight (periodic over m_eff cells) ---
    # Treat φ as living on m_eff cells; final row wraps to first.
    i_float = (phi / dphi) % m_eff                  # [0, m_eff)
    i0 = np.floor(i_float).astype(np.int64)         # 0 .. m_eff-1
    i1 = (i0 + 1) % m_eff                           # wrap
    t = i_float - i0                                # [0,1)

    # Gather corner values with advanced indexing (vectorized)
    u00 = grid[i0, j0]
    u01 = grid[i0, j1]
    u10 = grid[i1, j0]
    u11 = grid[i1, j1]

    # Bilinear blend
    u0 = (1.0 - s) * u00 + s * u01
    u1 = (1.0 - s) * u10 + s * u11
    return (1.0 - t) * u0 + t * u1


def advect_point_on_sphere(X, Y, Z, U, V, W, init_pos, num_steps=20, dt=0.1):
    """
    Advect a point on the sphere according to the vector field.

    Parameters
    ----------
    X, Y, Z : np.ndarray, shape (m, n)
        Cartesian coordinates of the spherical grid points.
        Typically produced by `generate_sphere_grid(...)`.
    U, V, W : np.ndarray, shape (m, n)
        Vector field components at the grid points, in Cartesian coordinates.
        Assumed to be (approximately) tangent to the sphere at (X,Y,Z).
    init_pos : (3,) or (k, 3)
        Initial Cartesian point(s) on the sphere. If (3,), advects one particle.
    num_steps : int, default=20
        Number of advection steps.
    dt : float, default=0.1
        Time step size.

    Returns
    -------
    pos : np.ndarray, shape (3,)
        Final position on the sphere after advection.
    """

    m, n = X.shape  # m rows (φ), n cols (θ)

    # --- Uniform axes ---
    # Theta[0, :] is θ-axis (0..π), Phi[:, 0] is φ-axis (0..2π)
    # With endpoints included, the seam is duplicated: use m_eff = m-1 for periodic interpolation.
    m_eff = m - 1

    dtheta = np.pi / (n - 1)        # spacing in θ (non-periodic)
    dphi   = 2.0 * np.pi / m_eff    # spacing in φ (periodic over m_eff)

    # init -> (k,3), remember if it was a single point
    P0_was_1d = (np.asarray(init_pos).ndim == 1)
    P = np.atleast_2d(np.asarray(init_pos, float))

    # preserve each particle’s radius and snap exactly to the sphere
    _, _, R0 = cart2sph_angles_batch(P)
    P = normalize_to_radius(P, R0)

    # RK2 advection loop (no inner functions)
    for _ in range(num_steps):
        th, ph, _ = cart2sph_angles_batch(P)
        ux = bilinear_sample_scalar(U, th, ph, dtheta, dphi, m_eff)
        vy = bilinear_sample_scalar(V, th, ph, dtheta, dphi, m_eff)
        wz = bilinear_sample_scalar(W, th, ph, dtheta, dphi, m_eff)
        V1 = np.stack([ux, vy, wz], axis=1)
        V1 = project_to_tangent(P, V1)
    
        mid = normalize_to_radius(P + 0.5 * dt * V1, R0)
    
        th_m, ph_m, _ = cart2sph_angles_batch(mid)
        ux = bilinear_sample_scalar(U, th_m, ph_m, dtheta, dphi, m_eff)
        vy = bilinear_sample_scalar(V, th_m, ph_m, dtheta, dphi, m_eff)
        wz = bilinear_sample_scalar(W, th_m, ph_m, dtheta, dphi, m_eff)
        V2 = np.stack([ux, vy, wz], axis=1)
        V2 = project_to_tangent(mid, V2)
    
        P = normalize_to_radius(P + dt * V2, R0)
        
    return P[0] if P0_was_1d else P


def create_training_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                         U: np.ndarray, V: np.ndarray, W: np.ndarray,
                         num_samples: int = 5000, num_steps: int = 20,
                         dt: float = 0.1, noise_level: float = 0.0,
                         save_particles_path: Optional[str] = None,
                         renormalize_after_noise: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training pairs (x0 -> xT) by advecting random sphere points under a fixed vector field.

    Pipeline (all vectorized):
      1) Sample `num_samples` initial points uniformly on S^2 (radius 1).
      2) Advect all points for `num_steps` of size `dt` via `advect_point_on_sphere`.
      3) Optionally add Gaussian noise to the final points.
      4) (Optionally) renormalize final points to radius 1 to stay on-manifold.
      5) Deterministic 80/20 split into train/test.

    Parameters
    ----------
    X, Y, Z : (m, n)
        Spherical grid coordinates (from `generate_sphere_grid`).
    U, V, W : (m, n)
        Cartesian components of the tangent vector field at each grid point.
    num_samples : int
        Number of (x0, xT) pairs to generate.
    num_steps : int
        Number of RK2 steps for advection.
    dt : float
        Time step for advection.
    noise_level : float
        Std. dev. of i.i.d. Gaussian noise added to final positions (Cartesian).
        Set 0.0 for noiseless trajectories.
    save_particles_path : str or None
        If given, saves the initial points to this .npy path (directories created).
    renormalize_after_noise : bool
        If True, reproject final points to the sphere after adding noise.

    Returns
    -------
    train_init, train_final, test_init, test_final : np.ndarray
        Arrays of shape (N_train, 3) / (N_test, 3) with the 80/20 split.
    """
    # sample initial points on S^2 (radius 1)
    initial_points = sample_random_points_on_sphere(num_samples, radius=1.0)

    # optional: persist canonical particles for downstream reproducibility
    if save_particles_path is not None:
        os.makedirs(os.path.dirname(save_particles_path), exist_ok=True)
        np.save(save_particles_path, initial_points)
        print(f"Saved {len(initial_points)} initial points as particles to {save_particles_path}")

    # advect all points at once (vectorized RK2)
    final_points = advect_point_on_sphere(X, Y, Z, U, V, W, init_pos=initial_points, num_steps=num_steps, dt=dt)

    # add noise (optional)
    if noise_level > 0.0:
        final_points = final_points + np.random.normal(scale=noise_level, size=final_points.shape)

    # snap back to the unit sphere (optional but usually desirable for on-manifold targets)
    if renormalize_after_noise:
        final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)

    # deterministic 80/20 split (keep order stable for reproducibility)
    train_size = int(0.8 * num_samples)
    train_init  = initial_points[:train_size]
    train_final = final_points[:train_size]
    test_init   = initial_points[train_size:]
    test_final  = final_points[train_size:]

    # Print some statistics
    print(f"Generated {num_samples} point pairs")
    print(f"Training set: {len(train_init)} pairs")
    print(f"Test set: {len(test_final)} pairs")
    print(f"Noise level: {noise_level}")
    print(f"Average initial norm: {np.mean(np.linalg.norm(initial_points, axis=1)):.6f}")
    print(f"Average final norm: {np.mean(np.linalg.norm(final_points, axis=1)):.6f}")
    
    return train_init, train_final, test_init, test_final

def create_sphere_with_vector_field(show_plot: bool = True, noise_level: float = 0.0, grid_size: int = 100, quiver_skip: int = 5, num_samples: int = 5000):
    """
    Build a spherical grid, generate a smooth tangent vector field on it,
    create (x0 -> xT) training pairs via advection, and optionally plot.

    Parameters
    ----------
    show_plot : bool
        If True, render a simple 3D visualization: sphere, sparse quiver, and
        a few example (x0 -> xT) pairs.
    noise_level : float
        Std. dev. of i.i.d. Gaussian noise added to final points inside
        `create_training_data` (set 0.0 for noiseless targets).
    grid_size : int
        Resolution per angular dimension for the sphere grid.
    quiver_skip : int
        Plot every `quiver_skip`-th vector in each grid dimension.
    num_samples : int, default=5000
        Number of (x0, xT) pairs to generate.

    Returns
    -------
    train_init, train_final, test_init, test_final : np.ndarray
        Arrays of shape (N_train, 3) / (N_test, 3) with the 80/20 split.
    """
    # --- Grid (unit sphere) ---
    X, Y, Z, Theta, Phi = generate_sphere_grid(grid_size=grid_size, radius=1.0)

    # --- Smooth random tangent field (low-order spherical harmonics–like) ---
    U, V, W = generate_random_continuous_vector_field_on_sphere(X, Y, Z, Theta, Phi)

    # --- Training data via vectorized advection (80/20 split inside) ---
    train_init, train_final, test_init, test_final = create_training_data(X, Y, Z, U, V, W, num_samples=num_samples, noise_level=noise_level)

    if show_plot:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Surface
        ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
        
        # Sparse vector field
        s = quiver_skip
        ax.quiver(X[::s, ::s], Y[::s, ::s], Z[::s, ::s],
                  U[::s, ::s], V[::s, ::s], W[::s, ::s],
                  length=0.1, color='C0', alpha=0.7)

        # Plot some example pairs
        for i in range(min(10, len(test_init))):
            # Initial point
            ax.scatter(test_init[i, 0], test_init[i, 1], test_init[i, 2], 
                      color='green', marker='o', s=50)
            # Final point
            ax.scatter(test_final[i, 0], test_final[i, 1], test_final[i, 2], 
                      color='red', marker='*', s=100)
            # Arrow from initial to final
            ax.quiver(test_init[i, 0], test_init[i, 1], test_init[i, 2],
                     test_final[i, 0] - test_init[i, 0],
                     test_final[i, 1] - test_init[i, 1],
                     test_final[i, 2] - test_init[i, 2],
                     color='black', alpha=0.5, arrow_length_ratio=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Sphere with Vector Field and Example Point Pairs')
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        plt.close()

    return train_init, train_final, test_init, test_final
