"""
MPS Compatibility Utilities.

Provides MPS-compatible implementations of operations that don't
have backward support on Metal (e.g., cdist).
"""

import torch
import torch.nn.functional as F


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between x and y.

    MPS-compatible alternative to torch.cdist.

    Args:
        x: (batch, n, d) or (n, d)
        y: (batch, m, d) or (m, d)

    Returns:
        distances: (batch, n, m) or (n, m)
    """
    # Ensure 3D
    x_3d = x.unsqueeze(0) if x.dim() == 2 else x
    y_3d = y.unsqueeze(0) if y.dim() == 2 else y

    # Compute squared distances using expansion
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    x_sq = (x_3d ** 2).sum(dim=-1, keepdim=True)  # (batch, n, 1)
    y_sq = (y_3d ** 2).sum(dim=-1, keepdim=True)  # (batch, m, 1)

    # (batch, n, m)
    dist_sq = x_sq + y_sq.transpose(-2, -1) - 2 * torch.bmm(x_3d, y_3d.transpose(-2, -1))

    # Clamp to avoid negative values from numerical errors
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq + 1e-12)  # Small epsilon for numerical stability

    # Return same shape as input
    if x.dim() == 2:
        return dist.squeeze(0)
    return dist


def pairwise_distances_self(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances within a single set of points.

    Args:
        x: (batch, n, d) or (n, d)

    Returns:
        distances: (batch, n, n) or (n, n)
    """
    return pairwise_distances(x, x)


def soft_nearest_neighbors(
    x: torch.Tensor,
    centers: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute soft assignment to nearest centers (MPS-compatible).

    Args:
        x: (batch, d) query points
        centers: (k, d) center points
        temperature: softmax temperature

    Returns:
        weights: (batch, k) soft assignments
    """
    # Compute distances
    dists = pairwise_distances(x.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)

    # Soft assignment via negative distance
    weights = F.softmax(-dists / temperature, dim=-1)

    return weights
