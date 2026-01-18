"""
Baseline Models WITHOUT DAT/H3 Knowledge.

These models serve as controls to measure whether DAT/H3 principles
provide any predictive advantage. Same parameter counts, same training,
just without the topological inductive biases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List


class StandardMLP(nn.Module):
    """
    Standard multi-layer perceptron baseline.
    No topological structure, just learned representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning dict for compatibility."""
        # Flatten if needed
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.view(B * L, D)
            features = self.backbone(x)
            pred = self.output_head(features)
            pred = pred.view(B, L, -1)
        else:
            features = self.backbone(x)
            pred = self.output_head(features)

        return {
            'prediction': pred,
            'features': features,
            # No topological outputs
            'parallel': None,
            'perp': None,
            'energy': None,
            'basin_probs': None,
            'shell_occupancy': None
        }


class StandardTransformer(nn.Module):
    """
    Standard transformer baseline for sequence data.
    Uses vanilla attention without quasicrystal modifications.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Standard positional encoding (sinusoidal)
        self.register_buffer('pos_encoding', self._build_pos_encoding())

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def _build_pos_encoding(self, max_len: int = 1024) -> torch.Tensor:
        """Standard sinusoidal positional encoding."""
        pe = torch.zeros(max_len, self.hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float() * (-math.log(10000.0) / self.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        is_sequence = x.dim() == 3

        if not is_sequence:
            # Add sequence dimension
            x = x.unsqueeze(1)

        B, L, D = x.shape

        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :L, :]

        # Transform
        features = self.transformer(x)

        # Output
        pred = self.output_head(features)

        if not is_sequence:
            pred = pred.squeeze(1)
            features = features.squeeze(1)

        return {
            'prediction': pred,
            'features': features,
            'parallel': None,
            'perp': None,
            'energy': None,
            'basin_probs': None,
            'shell_occupancy': None
        }


class StandardConvNet(nn.Module):
    """
    Standard 1D ConvNet baseline for sequential data.
    No icosahedral structure, just learned filters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Conv layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            ])
        self.convs = nn.Sequential(*layers)

        # Global pooling and output
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        is_sequence = x.dim() == 3

        if not is_sequence:
            x = x.unsqueeze(1)

        B, L, D = x.shape

        # Project
        x = self.input_proj(x)  # (B, L, hidden)

        # Conv expects (B, C, L)
        x = x.transpose(1, 2)
        features = self.convs(x)
        features = features.transpose(1, 2)  # Back to (B, L, hidden)

        # Output
        pred = self.output_head(features)

        if not is_sequence:
            pred = pred.squeeze(1)
            features = features.squeeze(1)

        return {
            'prediction': pred,
            'features': features,
            'parallel': None,
            'perp': None,
            'energy': None,
            'basin_probs': None,
            'shell_occupancy': None
        }


class MatchedBaseline(nn.Module):
    """
    Baseline model matched to DAT_H3_Predictor parameter count.
    Ensures fair comparison by having same capacity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        target_params: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Build backbone
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks to match capacity
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(ResidualBlock(hidden_dim))

        # Additional capacity layers if needed
        self.extra_layers = nn.ModuleList()

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Adjust to match target parameters
        if target_params is not None:
            self._match_parameters(target_params)

    def _match_parameters(self, target: int):
        """Add layers to match target parameter count."""
        current = sum(p.numel() for p in self.parameters())

        while current < target * 0.95:  # Within 5%
            self.extra_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            current = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        is_sequence = x.dim() == 3

        if is_sequence:
            B, L, D = x.shape
            x = x.view(B * L, D)

        # Process
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        for layer in self.extra_layers:
            x = F.gelu(layer(x)) + x

        features = x
        pred = self.output_head(features)

        if is_sequence:
            pred = pred.view(B, L, -1)
            features = features.view(B, L, -1)

        return {
            'prediction': pred,
            'features': features,
            'parallel': None,
            'perp': None,
            'energy': None,
            'basin_probs': None,
            'shell_occupancy': None
        }


class ResidualBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def create_matched_baseline(dat_h3_model: nn.Module, input_dim: int, output_dim: int = 1) -> MatchedBaseline:
    """
    Create a baseline model with same parameter count as DAT-H3 model.
    """
    target_params = sum(p.numel() for p in dat_h3_model.parameters())

    return MatchedBaseline(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=output_dim,
        num_layers=4,
        target_params=target_params
    )
