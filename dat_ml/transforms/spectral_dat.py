"""
Spectral DAT Transform.

Applies DAT principles in Fourier/frequency space, matching the
successful approach from weather skeleton analysis:
- Golden ratio (φ) frequency shells
- Learnable shell weights
- Spectral filtering before prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

TAU = (1 + math.sqrt(5)) / 2  # Golden ratio φ ≈ 1.618


class SpectralDATLayer(nn.Module):
    """
    Applies DAT principles in Fourier space.

    From your weather scripts:
    - FFT to frequency domain
    - Apply golden ratio shell mask
    - IFFT back to spatial domain

    This mirrors weather_bridge_final.py approach.
    """

    def __init__(
        self,
        anchor: float = 21.0,
        n_shells: int = 7,
        shell_range: Tuple[int, int] = (-3, 4),
        base_sigma: float = 10.0,
        learnable: bool = True
    ):
        super().__init__()
        self.anchor = anchor
        self.n_shells = n_shells
        self.shell_range = shell_range
        self.base_sigma = base_sigma

        # Shell targets at golden ratio intervals
        shell_targets = [anchor * (TAU ** n) for n in range(shell_range[0], shell_range[1])]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        if learnable:
            # Learnable shell weights (start at 1.0)
            self.shell_weights = nn.Parameter(torch.ones(len(shell_targets)))
            # Learnable sigma multipliers
            self.sigma_mult = nn.Parameter(torch.ones(len(shell_targets)))
            # Learnable low-freq preservation
            self.low_freq_weight = nn.Parameter(torch.tensor(0.6))
        else:
            self.register_buffer('shell_weights', torch.ones(len(shell_targets)))
            self.register_buffer('sigma_mult', torch.ones(len(shell_targets)))
            self.register_buffer('low_freq_weight', torch.tensor(0.6))

    def create_dat_mask(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Create golden ratio frequency mask."""
        # Handle 1D, 2D, or batched input
        if len(shape) == 1:
            nx = shape[0]
            kx = torch.fft.fftfreq(nx, device=device) * nx
            k_mag = torch.abs(kx)
        elif len(shape) == 2:
            ny, nx = shape
            kx = torch.fft.fftfreq(nx, device=device)
            ky = torch.fft.fftfreq(ny, device=device)
            KX, KY = torch.meshgrid(kx, ky, indexing='ij')
            k_mag = torch.sqrt(KX.T**2 + KY.T**2) * nx
        else:
            # For 1D sequence treated as features
            nx = shape[-1]
            kx = torch.fft.fftfreq(nx, device=device) * nx
            k_mag = torch.abs(kx)

        # Build mask from golden ratio shells
        dat_mask = torch.zeros_like(k_mag)

        for i, target in enumerate(self.shell_targets):
            sigma = self.base_sigma + (target * 0.15)
            sigma = sigma * F.softplus(self.sigma_mult[i])
            weight = F.softplus(self.shell_weights[i])
            dat_mask = dat_mask + weight * torch.exp(-((k_mag - target)**2) / (2 * sigma**2))

        # Low frequency preservation (like in your weather scripts)
        low_freq_thresh = 10.0
        low_freq_mask = (k_mag < low_freq_thresh).float()
        low_freq_val = torch.sigmoid(self.low_freq_weight)
        dat_mask = dat_mask + low_freq_mask * low_freq_val

        # Clamp to [0, 1]
        dat_mask = dat_mask.clamp(0, 1)

        return dat_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral DAT filtering.

        Args:
            x: Input tensor (batch, seq_len) or (batch, height, width)

        Returns:
            Filtered tensor with DAT frequency structure
        """
        orig_shape = x.shape

        # Handle different input shapes
        if x.dim() == 2:
            # (batch, features) -> 1D FFT per sample
            batch_size, seq_len = x.shape

            # FFT
            x_fft = torch.fft.fft(x, dim=-1)
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)

            # Create mask
            mask = self.create_dat_mask((seq_len,), x.device)

            # Apply mask
            x_filtered = x_fft_shifted * mask

            # IFFT
            x_out = torch.fft.ifft(torch.fft.ifftshift(x_filtered, dim=-1), dim=-1).real

        elif x.dim() == 3:
            # (batch, height, width) -> 2D FFT
            batch_size = x.shape[0]

            # FFT per sample
            x_fft = torch.fft.fft2(x)
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

            # Create mask
            mask = self.create_dat_mask(x.shape[1:], x.device)

            # Apply mask
            x_filtered = x_fft_shifted * mask

            # IFFT
            x_out = torch.fft.ifft2(torch.fft.ifftshift(x_filtered, dim=(-2, -1))).real

        else:
            # Flatten and treat as 1D
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_out = self.forward(x_flat).view(orig_shape)

        return x_out


class SpectralDATPredictor(nn.Module):
    """
    Neural network that applies spectral DAT filtering before prediction.

    This combines your weather skeleton approach with ML:
    1. Apply spectral DAT filtering (golden ratio frequencies)
    2. Extract features from filtered signal
    3. Predict target
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_shells: int = 7,
        anchor: float = 21.0,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual

        # Spectral DAT layer
        self.spectral_dat = SpectralDATLayer(
            anchor=anchor,
            n_shells=n_shells,
            learnable=True
        )

        # If using residual, we concatenate original + filtered
        effective_input = input_dim * 2 if use_residual else input_dim

        # Feature extraction MLP
        self.encoder = nn.Sequential(
            nn.Linear(effective_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with spectral DAT filtering."""
        # Flatten if needed
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.view(B, -1)

        # Apply spectral DAT filter
        x_filtered = self.spectral_dat(x)

        # Combine original and filtered (residual connection)
        if self.use_residual:
            x_combined = torch.cat([x, x_filtered], dim=-1)
        else:
            x_combined = x_filtered

        # Encode
        features = self.encoder(x_combined)

        # Predict
        prediction = self.head(features)

        return {
            'prediction': prediction,
            'features': features,
            'filtered': x_filtered,
            'parallel': None,  # No E6 projection in this model
            'perp': None,
            'energy': None,
            'shell_occupancy': None
        }


class SpectralBoostLayer(nn.Module):
    """
    'Bold' version from boldweather.py - boosts rather than masks.

    Instead of zeroing out non-DAT frequencies, this AMPLIFIES
    frequencies at golden ratio intervals while preserving everything.
    """

    def __init__(
        self,
        anchor: float = 21.0,
        n_shells: int = 7,
        shell_range: Tuple[int, int] = (-3, 4),
        base_sigma: float = 12.0,
        boost_factor: float = 5.0,
        learnable: bool = True
    ):
        super().__init__()
        self.anchor = anchor
        self.base_sigma = base_sigma

        shell_targets = [anchor * (TAU ** n) for n in range(shell_range[0], shell_range[1])]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        if learnable:
            self.boost_weights = nn.Parameter(torch.full((len(shell_targets),), boost_factor))
            self.sigma_mult = nn.Parameter(torch.ones(len(shell_targets)))
        else:
            self.register_buffer('boost_weights', torch.full((len(shell_targets),), boost_factor))
            self.register_buffer('sigma_mult', torch.ones(len(shell_targets)))

    def create_boost_mask(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Create boost mask (starts at 1.0, adds boosts at shell frequencies)."""
        if len(shape) == 1:
            nx = shape[0]
            kx = torch.fft.fftfreq(nx, device=device) * nx
            k_mag = torch.abs(kx)
        else:
            nx = shape[-1]
            kx = torch.fft.fftfreq(nx, device=device) * nx
            k_mag = torch.abs(kx)

        # Start with baseline of 1.0 (keep everything)
        boost_mask = torch.ones_like(k_mag)

        for i, target in enumerate(self.shell_targets):
            sigma = self.base_sigma * F.softplus(self.sigma_mult[i])
            boost = F.softplus(self.boost_weights[i])
            boost_mask = boost_mask + boost * torch.exp(-((k_mag - target)**2) / (2 * sigma**2))

        return boost_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral boost at golden ratio frequencies."""
        if x.dim() == 2:
            batch_size, seq_len = x.shape

            x_fft = torch.fft.fft(x, dim=-1)
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)

            boost_mask = self.create_boost_mask((seq_len,), x.device)

            x_boosted = x_fft_shifted * boost_mask

            x_out = torch.fft.ifft(torch.fft.ifftshift(x_boosted, dim=-1), dim=-1).real

        else:
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_out = self.forward(x_flat).view(x.shape)

        return x_out
