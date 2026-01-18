"""
DAT-ML: Golden Ratio Spectral Filtering

Drop-in PyTorch layer for φ-frequency filtering derived from H₃ manifold geometry.
Improves prediction on chaotic/turbulent systems (weather, Mackey-Glass, Lorenz).

Usage:
    from dat_ml import SpectralDATLayer

    layer = SpectralDATLayer()  # Theory-derived defaults
    x_filtered = layer(x)       # Apply golden ratio filtering

Theory:
    The depletion constant δ₀ = (√5-1)/4 ≈ 0.309 arises from icosahedral (H₃)
    symmetry. The optimal filter width satisfies: σ × δ₀ ≈ 1.081 (H₃ coordination).

References:
    - H3-Hybrid-Discovery: https://github.com/SolomonB14D3/H3-Hybrid-Discovery
    - Discrete-Alignment-Theory: https://github.com/SolomonB14D3/Discrete-Alignment-Theory
"""

from .spectral_dat_layer import (
    SpectralDATLayer,
    SpectralDATBlock,
    SpectralDATPredictor,
    add_spectral_dat,
    TAU,
    DELTA_0,
    H3_COORDINATION,
    OPTIMAL_SIGMA,
)

__version__ = '0.1.0'
__all__ = [
    'SpectralDATLayer',
    'SpectralDATBlock',
    'SpectralDATPredictor',
    'add_spectral_dat',
    'TAU',
    'DELTA_0',
    'H3_COORDINATION',
    'OPTIMAL_SIGMA',
]
