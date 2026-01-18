"""
H3 Hybrid Phase Transforms.

Implements concepts from H3-Hybrid-Discovery:
- Compression-based feature extraction (analogous to Plastic Flowering)
- Energy landscape modeling
- Coordination shell analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from ..core.mps_compat import pairwise_distances, pairwise_distances_self


class CompressionTransform(nn.Module):
    """
    Learnable compression transform inspired by Plastic Flowering.
    Progressively compresses feature space to find stable configurations.
    """

    def __init__(
        self,
        input_dim: int,
        compression_ratio: float = 1.081,  # From H3 RDF peak
        num_stages: int = 4,
        preserve_topology: bool = True
    ):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.num_stages = num_stages
        self.preserve_topology = preserve_topology

        # Progressive compression layers
        dims = [input_dim]
        for i in range(num_stages):
            next_dim = max(int(dims[-1] / compression_ratio), 6)
            dims.append(next_dim)

        self.compress_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(num_stages)
        ])

        # Skip connections for topology preservation
        if preserve_topology:
            self.skip_projs = nn.ModuleList([
                nn.Linear(dims[0], dims[i + 1])
                for i in range(num_stages)
            ])

        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Apply progressive compression.

        Returns:
            compressed: Final compressed representation
            intermediates: List of intermediate states (for analysis)
        """
        intermediates = [x]
        current = x
        identity = x

        for i, layer in enumerate(self.compress_layers):
            current = layer(current)
            current = F.gelu(current)

            if self.preserve_topology:
                # Add topology-preserving skip connection
                skip = self.skip_projs[i](identity)
                current = current + 0.1 * skip

            intermediates.append(current)

        return current, intermediates


class EnergyLandscape(nn.Module):
    """
    Models the energy landscape with H3 Hybrid characteristics.
    Creates a loss landscape biased toward the deep metastable minimum.
    """

    def __init__(
        self,
        feature_dim: int,
        target_energy: float = -7.68,  # H3 Hybrid energy
        baseline_energy: float = -6.48,  # FCC/HCP baseline
        num_basins: int = 8
    ):
        super().__init__()
        self.target_energy = target_energy
        self.baseline_energy = baseline_energy

        # Learnable basin centers (metastable states)
        self.basin_centers = nn.Parameter(torch.randn(num_basins, feature_dim) * 0.1)

        # Basin depths (energies)
        initial_depths = torch.linspace(baseline_energy, target_energy, num_basins)
        self.basin_depths = nn.Parameter(initial_depths)

        # Basin widths
        self.basin_widths = nn.Parameter(torch.ones(num_basins) * 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy for given features.

        Returns:
            energy: Scalar energy per sample
            basin_probs: Probability of being in each basin
        """
        # Compute distances to all basins (MPS-compatible)
        # x: (batch, features), centers: (basins, features)
        dists = pairwise_distances(x, self.basin_centers)

        # Gaussian basin contributions
        widths = F.softplus(self.basin_widths)
        basin_contributions = torch.exp(-0.5 * (dists / widths.unsqueeze(0)) ** 2)

        # Weighted energy
        basin_probs = F.softmax(-dists / widths.unsqueeze(0), dim=-1)
        energy = (basin_probs * self.basin_depths.unsqueeze(0)).sum(dim=-1)

        return energy, basin_probs


class CoordinationAnalysis(nn.Module):
    """
    Analyzes local coordination structure, inspired by RDF analysis.
    Identifies nearest-neighbor patterns in feature space.
    """

    def __init__(
        self,
        feature_dim: int,
        num_shells: int = 3,
        target_coordination: float = 1.081
    ):
        super().__init__()
        self.num_shells = num_shells
        self.target_coordination = target_coordination

        # Shell boundaries (learnable)
        shell_init = torch.linspace(0.5, 2.0, num_shells + 1)
        self.shell_boundaries = nn.Parameter(shell_init)

        # Shell embedding
        self.shell_embed = nn.Linear(num_shells, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute coordination features.

        Args:
            x: Batch of features (batch, features)

        Returns:
            coordination_features: Enhanced features with coordination info
            shell_occupancy: Occupancy of each coordination shell
        """
        batch_size = x.shape[0]

        if batch_size < 2:
            # Need multiple samples for coordination analysis
            shell_occupancy = torch.zeros(batch_size, self.num_shells, device=x.device)
            return x, shell_occupancy

        # Pairwise distances within batch (MPS-compatible)
        dists = pairwise_distances_self(x)

        # Mask self-distances
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
        dists = dists[mask].view(batch_size, batch_size - 1)

        # Count neighbors in each shell
        boundaries = F.softplus(self.shell_boundaries)
        shell_occupancy = torch.zeros(batch_size, self.num_shells, device=x.device)

        for i in range(self.num_shells):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            in_shell = ((dists >= lower) & (dists < upper)).float().sum(dim=-1)
            shell_occupancy[:, i] = in_shell

        # Normalize by expected coordination
        shell_occupancy = shell_occupancy / (self.target_coordination * (batch_size - 1) + 1e-8)

        # Embed shell info and add to features
        shell_features = self.shell_embed(shell_occupancy)
        coordination_features = x + shell_features

        return coordination_features, shell_occupancy


class H3HybridTransform(nn.Module):
    """
    Complete H3 Hybrid transform pipeline.
    Combines compression, energy landscape, and coordination analysis.
    """

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int = 32,
        num_compression_stages: int = 4,
        use_energy_landscape: bool = True,
        use_coordination: bool = True
    ):
        super().__init__()

        self.compression = CompressionTransform(
            input_dim,
            num_stages=num_compression_stages,
            preserve_topology=True
        )

        actual_compressed_dim = self.compression.output_dim

        self.use_energy_landscape = use_energy_landscape
        self.use_coordination = use_coordination

        if use_energy_landscape:
            self.energy = EnergyLandscape(actual_compressed_dim)

        if use_coordination:
            self.coordination = CoordinationAnalysis(actual_compressed_dim)

        self.output_dim = actual_compressed_dim

    def forward(self, x: torch.Tensor) -> dict:
        """
        Apply H3 Hybrid transform.

        Returns dict with:
            features: Transformed features
            energy: Energy values (if enabled)
            basin_probs: Basin probabilities (if enabled)
            shell_occupancy: Coordination shell occupancy (if enabled)
            intermediates: Compression intermediates
        """
        compressed, intermediates = self.compression(x)

        output = {
            'features': compressed,
            'intermediates': intermediates
        }

        if self.use_energy_landscape:
            energy, basin_probs = self.energy(compressed)
            output['energy'] = energy
            output['basin_probs'] = basin_probs

        if self.use_coordination:
            coord_features, shell_occ = self.coordination(compressed)
            output['features'] = coord_features
            output['shell_occupancy'] = shell_occ

        return output
