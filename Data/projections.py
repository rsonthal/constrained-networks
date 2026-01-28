import torch

def so3(X: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of flattened 3x3 matrices to SO(3) via SVD, enforce det=+1.

    Args:
        X: (batch_size, 9) tensor

    Returns:
        R_flat: (batch_size, 9) tensor
    """
    B = X.shape[0]
    M = X.view(B, 3, 3)

    U, _, Vh = torch.linalg.svd(M, full_matrices=False)  # Vh = V^T
    R = U @ Vh

    detR = torch.linalg.det(R)
    neg = detR < 0  # (B,)

    if neg.any():
        U = U.clone()
        U[neg, :, -1] *= -1
        R = U @ Vh

    return R.reshape(B, 9)