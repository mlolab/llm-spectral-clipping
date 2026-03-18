import torch
import math

def normalize_sigvals(G, steps: int):
    """
    Normalize singular values of G to approximately 1 using Newton-Schulz iterations.
    """

    if G.ndim == 1:
        return G / (G.norm() + 1e-7)

    return _normalize_sigvals_matrix(G, steps)

def clip_sigvals(G: torch.Tensor, clip_c: float = 1.0, ns_iter: int = 10):
    """
    Soft spectral clipping using the approximation:
        Hc(X) = (I + XX^T / c^2)^{-1/2} @ X

    This maps singular values σ → σ / sqrt(1 + σ²/c²), which:
        - Preserves small singular values (σ << c)
        - Clips large singular values toward c (σ >> c)

    Args:
        G: Input tensor (vector, matrix, or batched matrices)
        clip_c: Clipping threshold
        ns_iter: Number of Newton-Schulz iterations

    Returns:
        Spectrally clipped tensor with same shape as G
    """
    # vector input 
    if G.ndim == 1:
        norm = G.norm()
        if norm <= clip_c:
            return G
        return G / torch.sqrt(1 + (norm / clip_c) ** 2)

    X = G.bfloat16()

    # Work with the wide matrix (rows <= cols) for efficiency
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m = X.size(-2)  # number of rows (smaller dimension)
    XXT = X @ X.mT

    # Upper bound on max eigenvalue of XX^T (= max singular value squared of X)
    # using min of two bounds:
    #   1. Gershgorin: max absolute row sum
    #   2. Frobenius: ||XX^T||_F
    # For batched inputs, take max across all matrices (conservative bound)
    if X.ndim == 2:
        gershgorin_bound = XXT.abs().sum(dim=-1).max()
        frobenius_bound = XXT.norm()
    else:
        # Batched: compute per-matrix bounds, then take max across batch
        gershgorin_bound = XXT.abs().sum(dim=-1).max(dim=-1).values.max()
        frobenius_bound = XXT.norm(dim=(-2, -1)).max()
    s_max_XXT = torch.minimum(gershgorin_bound, frobenius_bound)

    # Compute (I + XX^T / c²)^{-1/2}
    # The eigenvalues of (I + XX^T / c²) are in [1, 1 + s_max_XXT/c²]
    # NOTE: We always compute the clipping (no early exit) to avoid GPU→CPU sync
    # It seems using the condition "if s_max_XXT.sqrt() <= clip_c" is even slower.
    I = torch.eye(m, dtype=X.dtype, device=X.device)
    A = I + XXT / (clip_c ** 2)
    # Keep alpha as a tensor to avoid GPU→CPU sync from .item()
    alpha = 1.0 + s_max_XXT / (clip_c ** 2)

    InvSqrt = matrix_inv_sqrt_NS(A, alpha, ns_iter=ns_iter)
    out = InvSqrt @ X

    if transposed:
        out = out.mT

    return out.to(G.dtype)


@torch.compile
def _normalize_sigvals_matrix(G, steps: int):
    """Implementation from Muon."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def _matrix_inv_sqrt_NS_core(Ahat: torch.Tensor, ns_iter: int = 10):
    """
    Newton-Schulz iteration for matrix inverse square root.

    Computes Ahat^{-1/2} where Ahat has eigenvalues in (0, 1].
    Supports both 2D (single matrix) and 3D+ (batched) inputs.

    Uses the coupled iteration:
        Y_{k+1} = 0.5 * Y_k @ (3I - Z_k @ Y_k)
        Z_{k+1} = 0.5 * (3I - Z_k @ Y_k) @ Z_k

    Converges: Y → Ahat^{1/2}, Z → Ahat^{-1/2}
    """
    device = Ahat.device
    n = Ahat.size(-1)  # Use last dimension for batched support

    # Create identity matrix (broadcasts over batch dimensions)
    I = torch.eye(n, dtype=Ahat.dtype, device=device)
    Y = Ahat.clone()
    Z = I.expand_as(Ahat).clone()

    for _ in range(ns_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    return Z


def matrix_inv_sqrt_NS(A: torch.Tensor, alpha, ns_iter: int = 10):
    """
    Compute A^{-1/2} using Newton-Schulz iteration with scaling.

    Args:
        A: Symmetric positive definite matrix (2D or batched 3D+)
        alpha: Upper bound on the largest eigenvalue of A (scalar tensor or float)
        ns_iter: Number of Newton-Schulz iterations

    Returns:
        Approximation of A^{-1/2}
    """
    assert A.size(-2) == A.size(-1), "A must be square in last two dimensions"

    # Scale so eigenvalues are in (0, 1]
    Ahat = A / alpha

    # Compute scaled inverse square root
    Z = _matrix_inv_sqrt_NS_core(Ahat, ns_iter)

    # Rescale - use torch.sqrt to avoid GPU→CPU sync when alpha is a tensor
    if isinstance(alpha, torch.Tensor):
        return Z / torch.sqrt(alpha)
    else:
        return Z / math.sqrt(alpha)



