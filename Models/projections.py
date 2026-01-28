import torch

def so3(X: torch.Tensor) -> torch.Tensor:
    """
    Project flattened 3x3 matrices to SO(3) via SVD, enforce det=+1.

    Works on:
      - (B, 9)
      - (B, S, 9)
      - (S, B, 9)  # transformer format [seq_len, batch, features]
      - (..., 9)

    Returns same shape as input.
    """
    *prefix, nine = X.shape
    assert nine == 9

    X = X.contiguous()

    M = X.reshape(-1, 3, 3)              # (N, 3, 3), N = prod(prefix)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh

    detR = torch.linalg.det(R)
    neg = detR < 0
    if neg.any():
        U = U.clone()
        U[neg, :, -1] *= -1
        R = U @ Vh

    return R.reshape(*prefix, 9)

# -----------------------
# Disk in R^2 (unit ball)
# -----------------------
def disk(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Project batch of 2D vectors onto the closed unit disk:
    if ||x||<=1 keep it, else scale to norm 1.

    Args:
        X: (B, 2)
    Returns:
        (B, 2)
    """
    n = torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(eps)
    scale = torch.minimum(torch.ones_like(n), 1.0 / n)
    return X * scale

# -----------------------
# Sphere S^{d-1} in R^d
# -----------------------
def sphere(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Project batch of vectors onto the unit sphere by normalization.

    Args:
        X: (B, d)
    Returns:
        (B, d) with ||x||=1 (up to eps)
    """
    n = torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(eps)
    return X / n

def se3(X: torch.Tensor) -> torch.Tensor:
    """
    Project flattened 4x4 matrices to SE(3) by:
      - projecting the top-left 3x3 block to SO(3) via SVD (det=+1),
      - keeping translation (top 3 entries of last column) as-is,
      - setting the last row to [0,0,0,1].

    Works on:
      - (B, 16)
      - (B, S, 16)
      - (..., 16)

    Assumes row-major flattening of the 4x4 matrix.
    Returns same shape as input.
    """
    *prefix, sixteen = X.shape
    assert sixteen == 16

    M = X.reshape(-1, 4, 4)          # (N, 4, 4)
    R = M[:, :3, :3]                 # (N, 3, 3)

    U, _, Vh = torch.linalg.svd(R, full_matrices=False)
    Rproj = U @ Vh

    detR = torch.linalg.det(Rproj)
    neg = detR < 0
    if neg.any():
        U = U.clone()
        U[neg, :, -1] *= -1
        Rproj = U @ Vh

    out = M.clone()
    out[:, :3, :3] = Rproj

    # enforce homogeneous last row
    out[:, 3, 0] = 0.0
    out[:, 3, 1] = 0.0
    out[:, 3, 2] = 0.0
    out[:, 3, 3] = 1.0

    return out.reshape(*prefix, 16)
