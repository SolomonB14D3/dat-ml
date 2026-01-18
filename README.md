# DAT-ML: Golden Ratio Spectral Filtering

A PyTorch layer that applies golden ratio (φ) frequency filtering derived from H₃ manifold geometry.

## Quick Start

```python
from dat_ml import SpectralDATLayer

# Add to any time series model
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.spectral = SpectralDATLayer()
        self.mlp = nn.Linear(input_dim * 2, hidden_dim)  # 2x for residual
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_filtered = self.spectral(x)
        x = torch.cat([x, x_filtered], dim=-1)
        return self.head(F.gelu(self.mlp(x)))
```

## Installation

```bash
pip install -e .
```

## The Theory

The depletion constant **δ₀ = (√5-1)/4 ≈ 0.309** arises from icosahedral (H₃) symmetry in the E₆ → H₃ Coxeter projection. This bounds vorticity growth in Navier-Stokes and appears in turbulence spectra.

**Key relationship discovered:**
```
σ × δ₀ = 3.498 × 0.309 = 1.081
```

This matches the H₃ coordination distance (1.081σ) from molecular dynamics simulations - **three independent sources converging**.

## Results

| Dataset | Improvement | p-value |
|---------|-------------|---------|
| ERA5 Weather (Z500) | +9.6% | 0.017 |
| Mackey-Glass τ=17 | +20% | <0.01 |
| Lorenz System | +3% | 0.04 |

**Works best on:** Moderate chaos, structured spectra, deterministic dynamics

**Not recommended for:** High chaos (τ=30), pure stochastic (exchange rates)

## Demo

```bash
python demo_spectral_dat.py
```

## References

- [H3-Hybrid-Discovery](https://github.com/SolomonB14D3/H3-Hybrid-Discovery)
- [Discrete-Alignment-Theory](https://github.com/SolomonB14D3/Discrete-Alignment-Theory)
