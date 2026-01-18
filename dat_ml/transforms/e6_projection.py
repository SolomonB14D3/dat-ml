"""
E6 Lattice Projection Transforms.

Implements the core mathematical machinery from Discrete Alignment Theory:
- E6 → H3 manifold projection
- Icosahedral symmetry operations
- Phason steering through 4D perpendicular rotations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math

# Golden ratio - fundamental to icosahedral symmetry
TAU = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895


def build_e6_basis() -> torch.Tensor:
    """
    Construct the E6 root lattice basis.
    E6 has 72 roots; we use a projection-friendly basis.
    """
    # E6 simple roots in 6D
    roots = torch.zeros(6, 6)

    # Standard E6 simple root system
    roots[0] = torch.tensor([1, -1, 0, 0, 0, 0], dtype=torch.float32)
    roots[1] = torch.tensor([0, 1, -1, 0, 0, 0], dtype=torch.float32)
    roots[2] = torch.tensor([0, 0, 1, -1, 0, 0], dtype=torch.float32)
    roots[3] = torch.tensor([0, 0, 0, 1, -1, 0], dtype=torch.float32)
    roots[4] = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.float32)
    roots[5] = torch.tensor([-0.5, -0.5, -0.5, -0.5, -0.5, math.sqrt(3)/2], dtype=torch.float32)

    return roots


def build_icosahedral_projection_matrix() -> torch.Tensor:
    """
    Build the 6D → 3D projection matrix preserving icosahedral symmetry.
    Projects E6 lattice onto the H3 manifold.
    """
    # Icosahedral projection uses golden ratio relationships
    # This projects 6D coordinates to 3D physical space
    proj = torch.zeros(3, 6, dtype=torch.float32)

    # Parallel space projection (physical 3D)
    proj[0] = torch.tensor([1, TAU, 0, -1, TAU, 0], dtype=torch.float32)
    proj[1] = torch.tensor([TAU, 0, 1, TAU, 0, -1], dtype=torch.float32)
    proj[2] = torch.tensor([0, 1, TAU, 0, -1, TAU], dtype=torch.float32)

    # Normalize
    proj = proj / torch.norm(proj, dim=1, keepdim=True)

    return proj


def build_perpendicular_projection_matrix() -> torch.Tensor:
    """
    Build the 6D → 3D perpendicular space projection.
    This captures the "phason" degrees of freedom used for topological steering.
    """
    proj = torch.zeros(3, 6, dtype=torch.float32)

    # Perpendicular space (phason space)
    proj[0] = torch.tensor([TAU, -1, 0, TAU, 1, 0], dtype=torch.float32)
    proj[1] = torch.tensor([-1, 0, TAU, 1, 0, TAU], dtype=torch.float32)
    proj[2] = torch.tensor([0, TAU, -1, 0, TAU, 1], dtype=torch.float32)

    proj = proj / torch.norm(proj, dim=1, keepdim=True)

    return proj


class E6Embedding(nn.Module):
    """
    Embed arbitrary dimensional data into E6 lattice space.
    """

    def __init__(self, input_dim: int, learnable: bool = True):
        super().__init__()
        self.input_dim = input_dim

        if learnable:
            # Learnable embedding to E6
            self.embed = nn.Linear(input_dim, 6, bias=False)
            # Initialize with structure-preserving weights
            nn.init.orthogonal_(self.embed.weight)
        else:
            # Fixed cyclic embedding
            self.register_buffer('embed_matrix', self._build_cyclic_embedding(input_dim))
            self.embed = lambda x: x @ self.embed_matrix.T

    def _build_cyclic_embedding(self, dim: int) -> torch.Tensor:
        """Build cyclic embedding matrix for fixed projection."""
        angles = torch.linspace(0, 2 * math.pi, dim + 1)[:-1]
        embed = torch.zeros(6, dim)
        for i in range(6):
            phase = i * math.pi / 3
            embed[i] = torch.cos(angles + phase)
        return embed.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input to E6 space."""
        return self.embed(x)


class H3Projection(nn.Module):
    """
    Project from E6 to H3 manifold with optional phason steering.
    """

    def __init__(self, phason_steering: bool = True, num_phason_steps: int = 64):
        super().__init__()

        self.register_buffer('proj_parallel', build_icosahedral_projection_matrix())
        self.register_buffer('proj_perp', build_perpendicular_projection_matrix())

        self.phason_steering = phason_steering
        self.num_phason_steps = num_phason_steps

        if phason_steering:
            # Learnable 4D rotation for phason steering
            # Parameterized as rotation angles in 4D perpendicular space
            self.phason_angles = nn.Parameter(torch.zeros(6))  # 6 rotation planes in 4D

    def _build_4d_rotation(self, angles: torch.Tensor) -> torch.Tensor:
        """Build 4D rotation matrix from 6 Euler-like angles."""
        # 4D has 6 independent rotation planes: xy, xz, xw, yz, yw, zw
        R = torch.eye(4, device=angles.device, dtype=angles.dtype)

        def rot_2d(theta):
            c, s = torch.cos(theta), torch.sin(theta)
            return c, s

        # Compose rotations for each plane
        planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for angle, (i, j) in zip(angles, planes):
            c, s = rot_2d(angle)
            Rp = torch.eye(4, device=angles.device, dtype=angles.dtype)
            Rp[i, i] = c
            Rp[j, j] = c
            Rp[i, j] = -s
            Rp[j, i] = s
            R = R @ Rp

        return R

    def forward(self, e6_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project E6 coordinates to H3 manifold.

        Returns:
            parallel: 3D physical space coordinates
            perpendicular: 3D phason space coordinates (optionally steered)
        """
        # Project to parallel (physical) space
        parallel = e6_coords @ self.proj_parallel.T

        # Project to perpendicular (phason) space
        perp = e6_coords @ self.proj_perp.T

        if self.phason_steering:
            # Extend to 4D for rotation, then project back
            perp_4d = torch.cat([perp, torch.zeros_like(perp[..., :1])], dim=-1)
            R = self._build_4d_rotation(self.phason_angles)
            perp_4d = perp_4d @ R.T
            perp = perp_4d[..., :3]

        return parallel, perp


class DATTransform(nn.Module):
    """
    Complete DAT transform pipeline: Input → E6 → H3 with phason steering.
    Produces topologically-informed features for downstream prediction.
    """

    def __init__(
        self,
        input_dim: int,
        learnable_embedding: bool = True,
        phason_steering: bool = True,
        output_combined: bool = True
    ):
        super().__init__()
        self.embedding = E6Embedding(input_dim, learnable=learnable_embedding)
        self.projection = H3Projection(phason_steering=phason_steering)
        self.output_combined = output_combined

        # Output dimension: parallel(3) + perp(3) or concatenated(6)
        self.output_dim = 6 if output_combined else 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input through DAT pipeline.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            H3 features (..., 6) if combined else (..., 3)
        """
        e6 = self.embedding(x)
        parallel, perp = self.projection(e6)

        if self.output_combined:
            return torch.cat([parallel, perp], dim=-1)
        return parallel


class PhasonLocalization(nn.Module):
    """
    Compute phason localization metric from DAT paper.
    Measures topological coherence of the projection.
    """

    def __init__(self, reference_cubic: bool = True):
        super().__init__()
        self.reference_cubic = reference_cubic
        # Baseline localization for cubic lattice
        self.register_buffer('cubic_baseline', torch.tensor(1.0))

    def forward(self, parallel: torch.Tensor, perp: torch.Tensor) -> torch.Tensor:
        """
        Compute localization factor.
        Target is ~4.2x improvement over cubic baseline (from DAT paper).
        """
        # Localization = variance ratio between parallel and perpendicular
        var_parallel = torch.var(parallel, dim=-1)
        var_perp = torch.var(perp, dim=-1)

        # Avoid division by zero
        localization = var_parallel / (var_perp + 1e-8)

        if self.reference_cubic:
            # Normalize to cubic baseline
            localization = localization / self.cubic_baseline

        return localization
