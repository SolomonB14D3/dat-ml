"""
H3-Inspired Neural Network Architecture.

Network topology reflects H3 manifold geometry:
- Icosahedral connectivity patterns
- Multi-scale feature aggregation (parallel + perpendicular)
- Topological skip connections preserving stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

from ..transforms.e6_projection import DATTransform, TAU
from ..transforms.h3_hybrid import H3HybridTransform


class IcosahedralBlock(nn.Module):
    """
    Network block with icosahedral connectivity.
    Uses 12-fold symmetry inspired by H3 manifold structure.
    """

    def __init__(self, dim: int, expansion: int = 12):
        super().__init__()
        self.dim = dim
        self.expansion = expansion

        # 12-way expansion (icosahedral vertices)
        self.expand = nn.Linear(dim, dim * expansion)

        # Pairwise interactions (edges of icosahedron = 30)
        self.interact = nn.Linear(dim * expansion, dim * expansion)

        # Contract back
        self.contract = nn.Linear(dim * expansion, dim)

        # Golden ratio scaling
        self.tau_scale = nn.Parameter(torch.tensor(TAU))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply icosahedral processing."""
        identity = x

        # Expand to 12 channels
        x = self.expand(x)
        x = F.gelu(x)

        # Interact
        x = self.interact(x)
        x = F.gelu(x)

        # Contract with golden ratio modulation
        x = self.contract(x) * self.tau_scale

        # Residual with normalization
        x = self.norm(x + identity)

        return x


class QuasicrystalAttention(nn.Module):
    """
    Attention mechanism inspired by quasicrystal structure.
    Uses aperiodic attention patterns based on golden ratio.
    """

    def __init__(self, dim: int, num_heads: int = 5):  # 5 = icosahedral rotation order
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Aperiodic position encoding based on golden ratio
        self.register_buffer('phi_phases', self._build_phi_phases())

    def _build_phi_phases(self, max_len: int = 1024) -> torch.Tensor:
        """Build golden-ratio-based position phases."""
        positions = torch.arange(max_len)
        # Fibonacci-like aperiodic phases
        phases = torch.zeros(max_len, self.num_heads)
        for h in range(self.num_heads):
            phases[:, h] = torch.cos(2 * math.pi * positions * (TAU ** h) / max_len)
        return phases

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply quasicrystal attention."""
        B, L, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Add aperiodic positional modulation
        phi = self.phi_phases[:L].unsqueeze(0).unsqueeze(-1)  # (1, L, heads, 1)
        q = q * (1 + 0.1 * phi)
        k = k * (1 + 0.1 * phi)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('blhd,bmhd->bhlm', q, k) * scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.einsum('bhlm,bmhd->blhd', attn, v)
        out = out.reshape(B, L, D)

        return self.proj(out)


class H3ManifoldLayer(nn.Module):
    """
    Single layer operating on H3 manifold representation.
    Processes parallel and perpendicular components with coupling.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Parallel space processing
        self.parallel_block = IcosahedralBlock(dim)

        # Perpendicular space processing
        self.perp_block = IcosahedralBlock(dim)

        # Cross-space coupling (phason-phonon interaction)
        self.coupling = nn.Bilinear(dim, dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        parallel: torch.Tensor,
        perp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process parallel and perpendicular components.

        Args:
            parallel: Physical space features (batch, dim)
            perp: Phason space features (batch, dim)

        Returns:
            Updated (parallel, perpendicular) tuple
        """
        # Process each space
        p_new = self.parallel_block(parallel)
        perp_new = self.perp_block(perp)

        # Couple the spaces (phason-phonon interaction)
        coupling = self.coupling(p_new, perp_new)

        # Add coupling to both
        p_out = self.norm(p_new + 0.1 * coupling)
        perp_out = self.norm(perp_new + 0.1 * coupling)

        return p_out, perp_out


class DAT_H3_Predictor(nn.Module):
    """
    Main prediction model combining DAT transforms with H3-inspired architecture.

    Architecture:
    1. Input → E6 embedding → H3 projection (parallel + perpendicular)
    2. Multi-layer H3 manifold processing with icosahedral blocks
    3. H3 Hybrid compression for final prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        use_attention: bool = True,
        use_h3_hybrid: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # DAT Transform: Input → E6 → H3
        self.dat_transform = DATTransform(
            input_dim,
            learnable_embedding=True,
            phason_steering=True,
            output_combined=False  # Keep parallel/perp separate
        )

        # Project to hidden dimension
        self.parallel_proj = nn.Linear(3, hidden_dim)
        self.perp_proj = nn.Linear(3, hidden_dim)

        # H3 Manifold layers
        self.h3_layers = nn.ModuleList([
            H3ManifoldLayer(hidden_dim)
            for _ in range(num_layers)
        ])

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = QuasicrystalAttention(hidden_dim * 2)

        # H3 Hybrid compression for output
        self.use_h3_hybrid = use_h3_hybrid
        if use_h3_hybrid:
            self.h3_hybrid = H3HybridTransform(
                hidden_dim * 2,
                use_energy_landscape=True,
                use_coordination=True
            )
            final_dim = self.h3_hybrid.output_dim
        else:
            final_dim = hidden_dim * 2

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Linear(final_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DAT-H3 predictor.

        Returns dict with:
            prediction: Main output prediction
            parallel: Parallel space features
            perp: Perpendicular space features
            energy: H3 Hybrid energy (if enabled)
            basin_probs: Basin probabilities (if enabled)
        """
        # Handle sequence input: (batch, seq, features) or (batch, features)
        is_sequence = x.dim() == 3
        if is_sequence:
            B, L, D = x.shape
            x_flat = x.view(B * L, D)
        else:
            x_flat = x
            B, L = x.shape[0], 1

        # DAT Transform
        e6 = self.dat_transform.embedding(x_flat)
        parallel, perp = self.dat_transform.projection(e6)

        # Project to hidden dim
        parallel = self.parallel_proj(parallel)
        perp = self.perp_proj(perp)

        # H3 Manifold processing
        for layer in self.h3_layers:
            parallel, perp = layer(parallel, perp)

        # Combine spaces
        combined = torch.cat([parallel, perp], dim=-1)

        # Reshape for attention if sequence
        if is_sequence and self.use_attention:
            combined = combined.view(B, L, -1)
            combined = self.attention(combined)
            combined = combined.view(B * L, -1)

        # H3 Hybrid transform
        output = {'parallel': parallel, 'perp': perp}

        if self.use_h3_hybrid:
            h3_out = self.h3_hybrid(combined)
            features = h3_out['features']
            output['energy'] = h3_out.get('energy')
            output['basin_probs'] = h3_out.get('basin_probs')
        else:
            features = combined

        # Prediction
        pred = self.output_head(features)

        if is_sequence:
            pred = pred.view(B, L, -1)

        output['prediction'] = pred
        return output


class EnsemblePredictor(nn.Module):
    """
    Ensemble of DAT-H3 predictors with different configurations.
    Combines multiple geometric perspectives.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 1
    ):
        super().__init__()

        self.predictors = nn.ModuleList([
            DAT_H3_Predictor(input_dim, hd, output_dim)
            for hd in hidden_dims
        ])

        # Learnable ensemble weights
        self.weights = nn.Parameter(torch.ones(len(hidden_dims)) / len(hidden_dims))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ensemble forward pass."""
        predictions = []
        energies = []

        for predictor in self.predictors:
            out = predictor(x)
            predictions.append(out['prediction'])
            if out.get('energy') is not None:
                energies.append(out['energy'])

        # Weighted combination
        weights = F.softmax(self.weights, dim=0)
        pred_stack = torch.stack(predictions, dim=0)
        ensemble_pred = (pred_stack * weights.view(-1, 1, 1)).sum(dim=0)

        output = {'prediction': ensemble_pred}

        if energies:
            energy_stack = torch.stack(energies, dim=0)
            output['energy'] = (energy_stack * weights.view(-1, 1)).sum(dim=0)

        return output
