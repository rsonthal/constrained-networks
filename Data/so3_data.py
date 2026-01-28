import numpy as np
import os
import pickle
from sphere_data import create_sphere_with_vector_field
from scipy.integrate import solve_ivp
from typing import List, Tuple

def so3_basis_matrices() -> List[np.ndarray]:
    """Return the three skew-symmetric generators B1, B2, B3 used in the ODE."""
    B1 = np.array([[0., -1., 0.],
                   [1.,  0., 0.],
                   [0.,  0., 0.]])
    B2 = np.array([[0., 0., 1.],
                   [0., 0., 0.],
                   [-1., 0., 0.]])
    B3 = np.array([[0., 0., 0.],
                   [0., 0., -1.],
                   [0., 1., 0.]])
    return [B1, B2, B3]

def so3_g(X: np.ndarray, Bs: List[np.ndarray]) -> np.ndarray:
    """g(X) = sum_i B_i @ X."""
    return Bs[0] @ X + Bs[1] @ X + Bs[2] @ X

def so3_ode_rhs(t: float, x_flat: np.ndarray, Bs: List[np.ndarray]) -> np.ndarray:
    """
    dX/dt = tr(X @ X + I) * g(X), with X reshaped from x_flat.
    (Kept verbatim with '@' as in your reference code.)
    """
    X = x_flat.reshape(3, 3)
    I = np.eye(3)
    tr = np.trace(X @ X + I)
    dX = tr * so3_g(X, Bs)
    return dX.ravel()

def proj_so3(X: np.ndarray) -> np.ndarray:
    """
    Project a N 3×3 matrix to SO(3) via SVD: X ≈ U V^T, enforce det=+1.
    """
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    U[neg, :, -1] *= -1
    return U @ Vt

def random_so3() -> np.ndarray:
    """
    Random element of SO(3) via QR on a Gaussian matrix with det=+1.
    """
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q

def random_so3_batch(N: int) -> np.ndarray:
    """
    N random elements of SO(3) via QR on a Gaussian matrix with det=+1.
    """
    A = np.random.randn(N, 3, 3)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    U[neg, :, -1] *= -1
    return U @ Vt  # (N,3,3)

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


def generate_so3_trajectory(t0: float = 0.0, t1: float = 10.0, 
                            num_steps: int = 11, noise_level: float = 0.0) -> List[np.ndarray]:
    """
    Integrate the SO(3) ODE from a random X(0) ∈ SO(3), collect states at t_eval,
    optionally add Gaussian noise, then project each back to SO(3).
    Returns a list of flattened 9D vectors [X(t0).ravel(), ..., X(t1).ravel()].
    """

    # Generate Basis
    Bs = so3_basis_matrices() # List of size 3, each matrix is 3 x 3

    X0 = random_so3()
    t_eval = np.linspace(t0, t1, num_steps)
    sol = solve_ivp(lambda t, x: so3_ode_rhs(t, x, Bs),
                    (t0, t1), X0.ravel(), t_eval=t_eval, method='RK45')

    # (num_steps, 3, 3)
    Xs = sol.y.T.reshape(num_steps, 3, 3)

    if noise_level > 0.0:
        norms = np.linalg.norm(Xs, axis=(1, 2), keepdims=True)              # (num_steps,1,1)
        Xs = Xs + noise_level * norms * np.random.randn(*Xs.shape)

    # Batched projection to SO(3)
    Xs = proj_so3(Xs)  # (num_steps,3,3)

    return Xs.reshape(num_steps, 9).tolist()

def generate_so3_dataset(num_samples: int = 5000, t0: float = 0.0, t1: float = 10.0, num_steps: int = 11, noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an SO(3) dataset by integrating the ODE
        dX/dt = tr(X @ X + I) * Σ_i B_i X,
    projecting back to SO(3) at each saved step.

    Returns 80/20 train/test splits of flattened 9D matrices:
        (train_init, train_final, test_init, test_final), each of shape (N, 9).

    Parameters
    ----------
    num_samples : int
        Number of pairs to generate.
    t0, t1 : float
        Time window for the ODE.
    num_steps : int
        Number of saved evaluation steps between t0 and t1 (inclusive).
        Initial = step 0, final = step num_steps-1.
    noise_level : float
        Additive Gaussian noise strength applied to each saved X(t) *before*
        projection back to SO(3).
    """
    print(f"Generating SO(3) dataset with {num_samples} samples "
          f"(t∈[{t0},{t1}], steps={num_steps}, noise={noise_level}) ...")

    initial_so3 = []
    final_so3 = []

    for _ in range(num_samples):
        traj = generate_so3_trajectory(t0=t0, t1=t1, num_steps=num_steps,
                                       noise_level=noise_level)
        initial_so3.append(traj[0])                 # X(t0)
        final_so3.append(traj[num_steps - 1])       # X(t1)

    initial_so3 = np.asarray(initial_so3)  # (N, 9)
    final_so3   = np.asarray(final_so3)    # (N, 9)

    # 80/20 split (deterministic)
    Ntr = int(0.8 * num_samples)
    train_init, test_init   = initial_so3[:Ntr], initial_so3[Ntr:]
    train_final, test_final = final_so3[:Ntr],   final_so3[Ntr:]

    print(f"Generated {num_samples} SO(3) pairs.")
    print(f"Train: {len(train_init)}   Test: {len(test_init)}")
    print(f"Test set: {len(test_final)} pairs")


    # quick diagnostics
    check_so3_constraints(train_init, "Initial (train)")
    check_so3_constraints(train_final, "Final   (train)")

    return train_init, train_final, test_init, test_final