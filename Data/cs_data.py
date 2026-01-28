import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional
from tqdm.auto import tqdm

# ============================================================================
# 1. SO(3) Math Helpers (From your CS Script)
# ============================================================================

def hat(a: np.ndarray) -> np.ndarray:
    """Skew-symmetric map for a vector in R^3."""
    ax, ay, az = a
    return np.array([[0, -az, ay],
                     [az, 0, -ax],
                     [-ay, ax, 0]], dtype=float)

def expSO3(a: np.ndarray) -> np.ndarray:
    """Compute exp(hat(a)) on SO(3)."""
    A = hat(a)
    return expm(A)

def logSO3(R: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute the Riemannian log map on SO(3).
    Returns (theta*n, theta, n).
    """
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = 0.5 * (tr - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3), 0.0, np.array([1.0, 0.0, 0.0])
    A = (R - R.T) / (2.0 * np.sin(theta))
    n = np.array([A[2, 1], A[0, 2], A[1, 0]])
    return theta * n, theta, n

def proj_so3(X: np.ndarray) -> np.ndarray:
    """
    Project a single 3x3 matrix to SO(3) via SVD.
    Used for applying noise while staying on manifold.
    """
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def proj_so3_batch(Xs: np.ndarray) -> np.ndarray:
    """
    Batched projection for (N, 3, 3) arrays.
    """
    U, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    U[neg, :, -1] *= -1
    return U @ Vt

# ============================================================================
# 2. Cucker-Smale Integrator Logic
# ============================================================================

def phi_cos_half(theta: float) -> float:
    """Communication weight function."""
    return np.cos(0.5 * theta) if theta < np.pi else 0.0

def cs_rhs_a(R_list: List[np.ndarray], a: np.ndarray, kappa: float, phi_fn) -> np.ndarray:
    """Compute the RHS for the angular velocity update (Lie algebra)."""
    N = len(R_list)
    rhs = np.zeros_like(a)
    for i in range(N):
        Ri = R_list[i]
        acc_i = np.zeros(3)
        for k in range(N):
            # Relative rotation R_ki = R_k^T R_i (note the transpose order matters for left/right invariant metrics)
            # Based on the snippet provided: Rki = R_list[k].T @ Ri
            Rki = R_list[k].T @ Ri
            u, theta, n = logSO3(Rki)
            if theta >= np.pi - 1e-12:
                continue
            phi = phi_fn(theta)
            if phi == 0.0:
                continue
            
            ak = a[k]
            c2 = np.cos(0.5 * theta)
            s2 = np.sin(0.5 * theta)
            
            # Cucker-Smale alignment term on sphere/SO(3)
            term = (1.0 - c2) * (np.dot(n, ak) * n) + s2 * np.cross(ak, n) + c2 * ak - a[i]
            acc_i += phi * term
        rhs[i] = (kappa / N) * acc_i
    return rhs

def step_rkmk2(R_list: List[np.ndarray], a: np.ndarray, dt: float, kappa: float, phi_fn=phi_cos_half):
    """Run one step of the Runge-Kutta-Munthe-Kaas (RKMK) integrator."""
    k1 = cs_rhs_a(R_list, a, kappa, phi_fn)
    
    # Intermediate step (half step)
    R_half = [R @ expSO3(0.5 * dt * a_i) for R, a_i in zip(R_list, a)]
    a_half = a + 0.5 * dt * k1
    
    # Full step calc
    k2 = cs_rhs_a(R_half, a_half, kappa, phi_fn)
    
    # Update
    R_next = [R @ expSO3(dt * a_h) for R, a_h in zip(R_list, a_half)]
    a_next = a + dt * k2
    
    return R_next, a_next

# ============================================================================
# 3. Trajectory & Dataset Generation
# ============================================================================

def generate_cs_flock_trajectory(flock_size: int, t0: float = 0.0, t1: float = 10.0,
                                 num_steps: int = 11, noise_level: float = 0.0,
                                 kappa: float = 1.0, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates a single Cucker-Smale flock.
    
    Returns
    -------
    init_flat : (flock_size, 9)
    final_flat: (flock_size, 9)
    """
    T = float(t1 - t0)
    dt = T / max(1, (num_steps - 1))

    rng = np.random.default_rng()
    
    # --- Initialization ---
    # Random rotations
    R = []
    for _ in range(flock_size):
        # Uniform-ish sampling (axis-angle with random axis)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 0.6 * np.pi * rng.random() # Restricted sector init per your snippet logic
        R.append(expSO3(angle * axis))
    
    # Random angular velocities
    a = rng.uniform(-1, 1, size=(flock_size, 3))
    
    def flatten_R(Rs: list[np.ndarray]) -> np.ndarray:
        stack = np.asarray(Rs, dtype=float)  # (flock_size, 3, 3)
        if noise_level > 0.0:
            stack = stack + noise_level * rng.normal(size=stack.shape)
            stack = proj_so3_batch(stack)
        return stack.reshape(flock_size, 9).astype(dtype, copy=False)

    init_flat = flatten_R(R)

    current_R, current_a = R, a
    for _ in range(num_steps - 1):
        current_R, current_a = step_rkmk2(current_R, current_a, dt, kappa)

    final_flat = flatten_R(current_R)
    return init_flat, final_flat

def generate_cs_dataset(num_samples: int = 5000, 
                        flock_size: int = 10,
                        t0: float = 0.0, 
                        t1: float = 8.0, 
                        num_steps: int = 11, 
                        noise_level: float = 0.0,
                        kappa: float = 1.0,
                        seed: Optional[int] = 0,
                        dtype=np.float32,
                        show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Cucker-Smale SO(3) dataset.
    
    Unlike the independent particle script, this runs simulations of flocks.
    It aggregates individual particle trajectories from multiple flock simulations 
    to reach the total `num_samples`.

    Returns 80/20 train/test splits of flattened 9D matrices:
       (train_init, train_final, test_init, test_final), each of shape (num_samples, flock_size, 9).
    """
    print(
        f"Generating CS SO(3) dataset with {num_samples} flocks "
        f"(flock_size={flock_size}, dim=9, t∈[{t0},{t1}], num_steps={num_steps}, "
        f"noise={noise_level}, kappa={kappa})"
    )

    rng = np.random.default_rng(seed)

    # Preallocate
    init = np.empty((num_samples, flock_size, 9), dtype=dtype)
    final = np.empty((num_samples, flock_size, 9), dtype=dtype)

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating flocks", unit="flock")
    
    for s in iterator:
        init_s, final_s = generate_cs_flock_trajectory(
            flock_size=flock_size,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            noise_level=noise_level,
            kappa=kappa,     
            dtype=dtype,
        )
        init[s] = init_s
        final[s] = final_s

    Ntr = int(0.8 * num_samples)
    train_init, test_init = init[:Ntr], init[Ntr:]
    train_final, test_final = final[:Ntr], final[Ntr:]

    print(f"Generated {num_samples} flock samples.")
    print(f"Train: {train_init.shape[0]}   Test: {test_init.shape[0]}")

    # diagnostics on a subset of particles (flatten to (N,9))
    check_so3_constraints(train_init.reshape(-1, 9), "Initial (train, flattened)")
    check_so3_constraints(train_final.reshape(-1, 9), "Final   (train, flattened)")

    return train_init, train_final, test_init, test_final

# ============================================================================
# 4. Diagnostics (Verbatim from first script)
# ============================================================================

def check_so3_constraints(so3_flat: np.ndarray, name: str) -> None:
    """
    Print SO(3) diagnostics: determinants and Frobenius norms of (R^T R - I).
    so3_flat: (N, 9) array of flattened 3×3 rotation matrices.
    """
    Rs = so3_flat.reshape(-1, 3, 3)        # (N, 3, 3)
    dets = np.linalg.det(Rs)               # (N,)

    # Compute R^T R for each matrix (vectorized)
    Rt = np.transpose(Rs, (0, 2, 1))       # (N, 3, 3)
    RtR = Rt @ Rs                           # (N, 3, 3)

    # Frobenius norm ||R^T R - I||_F per sample
    I = np.eye(3)
    errs = np.linalg.norm(RtR - I, axis=(1, 2))  # (N,)

    print(f"{name} - det mean={dets.mean():.6f}, det std={dets.std():.6f}")
    print(f"{name} - ||R^T R - I|| mean={errs.mean():.6f}, std={errs.std():.6f}")

    valid = (np.abs(dets - 1.0) < 1e-6) & (errs < 1e-6)
    print(f"{name} - valid SO(3): {valid.sum()}/{len(so3_flat)} ({100*valid.mean():.1f}%)")

# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    # Generate data
    train_i, train_f, test_i, test_f = generate_cs_dataset(
        num_samples=1000, 
        flock_size=10,
        t0=0.0, 
        t1=1.0,      # Short time for demo
        num_steps=5, # Coarse steps
        noise_level=0.001
    )