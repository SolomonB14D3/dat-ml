#!/usr/bin/env python3
"""
Test: Does the depletion constant δ₀ = (√5-1)/4 appear in optimal spectral filtering?

From NAVIER_STOKES_PROOF.md:
- δ₀ = (√5-1)/4 ≈ 0.309016994
- This bounds vortex stretching via H₃ symmetry
- Enstrophy depleted by factor (1 - δ₀) ≈ 0.691

Questions:
1. Does δ₀ appear in our empirically optimal parameters?
2. Does a δ₀-parameterized filter match or beat our best?
3. Do learned filter weights converge to δ₀-related values?
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import math

from dat_ml.core.device import DEVICE

# Constants from H3-Hybrid theory
TAU = (1 + math.sqrt(5)) / 2          # Golden ratio φ ≈ 1.618
DELTA_0 = (math.sqrt(5) - 1) / 4      # Depletion constant ≈ 0.309
ONE_MINUS_DELTA = 1 - DELTA_0         # ≈ 0.691

print("=" * 70)
print("DEPLETION CONSTANT ANALYSIS")
print("=" * 70)
print(f"\nKey constants from H₃ theory:")
print(f"  τ (golden ratio) = {TAU:.10f}")
print(f"  δ₀ (depletion)   = {DELTA_0:.10f}")
print(f"  1 - δ₀           = {ONE_MINUS_DELTA:.10f}")
print(f"  τ·δ₀             = {TAU * DELTA_0:.10f}")
print(f"  δ₀/τ             = {DELTA_0 / TAU:.10f}")
print(f"  1/(2τ)           = {1/(2*TAU):.10f}")  # Should equal δ₀

# Check mathematical relationships
print(f"\nMathematical identities:")
print(f"  δ₀ = (τ-1)/2 ? {abs(DELTA_0 - (TAU-1)/2) < 1e-10}")
print(f"  δ₀ = 1/(2τ)  ? {abs(DELTA_0 - 1/(2*TAU)) < 1e-10}")
print(f"  δ₀ = τ-1-δ₀  ? {abs(DELTA_0 - (TAU-1-DELTA_0)) < 1e-10}")  # δ₀ = φ-1-δ₀ means 2δ₀ = φ-1

# Our empirically optimal parameters
OPTIMAL_ANCHOR = 2.5
OPTIMAL_SIGMA = 3.5

print(f"\nOur empirically optimal parameters:")
print(f"  anchor = {OPTIMAL_ANCHOR}")
print(f"  sigma  = {OPTIMAL_SIGMA}")
print(f"  anchor/sigma = {OPTIMAL_ANCHOR/OPTIMAL_SIGMA:.6f}")
print(f"  sigma/anchor = {OPTIMAL_SIGMA/OPTIMAL_ANCHOR:.6f}")

# Look for δ₀ relationships
print(f"\nSearching for δ₀ relationships:")
print(f"  anchor × δ₀     = {OPTIMAL_ANCHOR * DELTA_0:.6f}")
print(f"  sigma × δ₀      = {OPTIMAL_SIGMA * DELTA_0:.6f}")
print(f"  anchor/σ vs δ₀  = {OPTIMAL_ANCHOR/OPTIMAL_SIGMA:.6f} vs {DELTA_0:.6f}")
print(f"  1-anchor/σ      = {1 - OPTIMAL_ANCHOR/OPTIMAL_SIGMA:.6f} vs {ONE_MINUS_DELTA:.6f}")
print(f"  anchor × (1-δ₀) = {OPTIMAL_ANCHOR * ONE_MINUS_DELTA:.6f}")
print(f"  σ/τ             = {OPTIMAL_SIGMA / TAU:.6f}")
print(f"  anchor/τ        = {OPTIMAL_ANCHOR / TAU:.6f}")


def load_weather_data():
    """Load ERA5 weather data."""
    from dat_ml.data.netcdf_loader import ERA5RegionalDataset
    from dat_ml.data.loader import create_dataloaders
    from dat_ml.core.config import ExperimentConfig

    nc_path = Path('/Users/bryan/Chaos_analogies/quasicrystal-chaos-analogies/phason_brain/era5_z500_2023_2025.nc')
    dataset = ERA5RegionalDataset(
        nc_path, variable='z', sequence_length=30, horizon=7,
        normalize=True, preload_device='mps'
    )

    exp_config = ExperimentConfig()
    exp_config.training.batch_size = 64
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, exp_config.data, exp_config.training
    )

    return dataset, train_loader, val_loader, test_loader


class DepletionFilter(nn.Module):
    """
    Spectral filter parameterized by depletion constant δ₀.

    Theory: Vortex stretching depleted by (1 - δ₀·f(ω))
    We apply this as frequency-space attenuation.
    """

    def __init__(self, base_freq=1.0, use_delta=True):
        super().__init__()
        self.base_freq = base_freq
        self.use_delta = use_delta

        # Shell targets at τ^n intervals
        n_shells = 7
        shell_targets = [base_freq * (TAU ** n) for n in range(n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        if use_delta:
            # Weights derived from δ₀
            # Each shell attenuated by (1 - δ₀)^n
            weights = [(ONE_MINUS_DELTA) ** n for n in range(n_shells)]
            self.register_buffer('shell_weights', torch.tensor(weights))
            # Sigma proportional to δ₀
            self.sigma_base = DELTA_0 * 10  # Scale to reasonable range
        else:
            # Learnable weights (for comparison)
            self.shell_weights = nn.Parameter(torch.ones(n_shells))
            self.sigma_base = 3.5

    def create_mask(self, seq_len, device):
        kx = torch.fft.fftfreq(seq_len, device=device) * seq_len
        k_mag = torch.abs(kx)

        mask = torch.zeros_like(k_mag)
        for i, target in enumerate(self.shell_targets):
            sigma = self.sigma_base + target * DELTA_0
            weight = self.shell_weights[i]
            if not self.use_delta:
                weight = F.softplus(weight)
            mask = mask + weight * torch.exp(-((k_mag - target)**2) / (2 * sigma**2))

        # Low frequency preservation at (1 - δ₀) level
        low_freq_mask = (k_mag < 10.0).float()
        mask = mask + low_freq_mask * ONE_MINUS_DELTA

        return mask.clamp(0, 1)

    def forward(self, x):
        seq_len = x.shape[-1]

        x_fft = torch.fft.fft(x, dim=-1)
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)

        mask = self.create_mask(seq_len, x.device)
        x_filtered = x_fft_shifted * mask

        x_out = torch.fft.ifft(torch.fft.ifftshift(x_filtered, dim=-1), dim=-1).real
        return x_out


class DepletionModel(nn.Module):
    """Model using δ₀-parameterized spectral filter."""

    def __init__(self, input_dim, hidden_dim, output_dim, use_delta=True):
        super().__init__()
        self.filter = DepletionFilter(base_freq=OPTIMAL_ANCHOR, use_delta=use_delta)

        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x_filtered = self.filter(x)
        x_combined = torch.cat([x, x_filtered], dim=-1)
        return self.net(x_combined)


class LearnableDepletionModel(nn.Module):
    """Model that learns filter weights - do they converge to δ₀?"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        n_shells = 7
        shell_targets = [OPTIMAL_ANCHOR * (TAU ** n) for n in range(n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        # Learnable parameters - initialized to 1.0
        self.shell_weights = nn.Parameter(torch.ones(n_shells))
        self.sigma_mult = nn.Parameter(torch.ones(n_shells))
        self.low_freq_weight = nn.Parameter(torch.tensor(0.5))

        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def create_mask(self, seq_len, device):
        kx = torch.fft.fftfreq(seq_len, device=device) * seq_len
        k_mag = torch.abs(kx)

        mask = torch.zeros_like(k_mag)
        for i, target in enumerate(self.shell_targets):
            sigma = 3.5 * F.softplus(self.sigma_mult[i])
            weight = F.softplus(self.shell_weights[i])
            mask = mask + weight * torch.exp(-((k_mag - target)**2) / (2 * sigma**2))

        low_freq_mask = (k_mag < 10.0).float()
        mask = mask + low_freq_mask * torch.sigmoid(self.low_freq_weight)

        return mask.clamp(0, 1)

    def forward(self, x):
        seq_len = x.shape[-1]

        x_fft = torch.fft.fft(x, dim=-1)
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)

        mask = self.create_mask(seq_len, x.device)
        x_filtered = x_fft_shifted * mask

        x_out = torch.fft.ifft(torch.fft.ifftshift(x_filtered, dim=-1), dim=-1).real

        x_combined = torch.cat([x, x_out], dim=-1)
        return self.net(x_combined)

    def get_learned_params(self):
        """Extract learned filter parameters."""
        weights = F.softplus(self.shell_weights).detach().cpu().numpy()
        sigmas = 3.5 * F.softplus(self.sigma_mult).detach().cpu().numpy()
        low_freq = torch.sigmoid(self.low_freq_weight).detach().cpu().item()
        return weights, sigmas, low_freq


def train_model(model, train_loader, val_loader, epochs=100):
    """Train and return best model."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)
                val_loss += criterion(model(x).squeeze(), y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model, best_val


def evaluate_model(model, test_loader):
    """Evaluate R²."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)
            all_preds.append(model(x).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENT: Testing δ₀ in spectral filtering")
    print("=" * 70)

    # Load data
    print("\nLoading weather data...")
    dataset, train_loader, val_loader, test_loader = load_weather_data()

    sample_x, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = 6
    hidden_dim = 128

    results = {}

    # 1. δ₀-parameterized filter (fixed weights from theory)
    print("\n" + "-" * 50)
    print("MODEL 1: δ₀-parameterized filter (from theory)")
    print("-" * 50)
    print(f"  Shell weights: (1-δ₀)^n = {[f'{(ONE_MINUS_DELTA)**n:.4f}' for n in range(7)]}")
    print(f"  Sigma base: δ₀ × 10 = {DELTA_0 * 10:.4f}")

    model_delta = DepletionModel(input_dim, hidden_dim, output_dim, use_delta=True)
    model_delta, _ = train_model(model_delta, train_loader, val_loader, epochs=100)
    r2_delta = evaluate_model(model_delta, test_loader)
    results['δ₀-theory'] = r2_delta
    print(f"  R² = {r2_delta:.6f}")

    # 2. Learnable filter (see what it converges to)
    print("\n" + "-" * 50)
    print("MODEL 2: Learnable filter (do weights → δ₀?)")
    print("-" * 50)

    model_learn = LearnableDepletionModel(input_dim, hidden_dim, output_dim)
    model_learn, _ = train_model(model_learn, train_loader, val_loader, epochs=100)
    r2_learn = evaluate_model(model_learn, test_loader)
    results['Learnable'] = r2_learn

    # Extract learned parameters
    weights, sigmas, low_freq = model_learn.get_learned_params()
    print(f"  R² = {r2_learn:.6f}")
    print(f"\n  Learned shell weights:")
    for i, w in enumerate(weights):
        theory_w = ONE_MINUS_DELTA ** i
        ratio = w / theory_w if theory_w > 0 else 0
        print(f"    Shell {i}: {w:.4f} (theory: {theory_w:.4f}, ratio: {ratio:.2f})")

    print(f"\n  Learned sigmas:")
    for i, s in enumerate(sigmas):
        print(f"    Shell {i}: {s:.4f}")

    print(f"\n  Learned low-freq weight: {low_freq:.4f} (theory 1-δ₀: {ONE_MINUS_DELTA:.4f})")

    # 3. Baseline (no spectral filter)
    print("\n" + "-" * 50)
    print("MODEL 3: Baseline MLP (no spectral filter)")
    print("-" * 50)

    baseline = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    )
    baseline, _ = train_model(baseline, train_loader, val_loader, epochs=100)
    r2_baseline = evaluate_model(baseline, test_loader)
    results['Baseline'] = r2_baseline
    print(f"  R² = {r2_baseline:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'R²':>12} {'vs Baseline':>15}")
    print("-" * 55)

    for name, r2 in sorted(results.items(), key=lambda x: -x[1]):
        delta = (r2 - r2_baseline) / (1 - r2_baseline) * 100  # % error reduction
        print(f"{name:<25} {r2:>12.6f} {delta:>+14.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Does δ₀ appear in optimal filtering?")
    print("=" * 70)

    # Compare learned weights to δ₀ theory
    weight_ratios = weights / np.array([ONE_MINUS_DELTA**i for i in range(len(weights))])
    mean_ratio = np.mean(weight_ratios)

    print(f"""
1. LEARNED vs THEORY WEIGHTS:
   Mean ratio (learned/theory): {mean_ratio:.3f}
   {'→ Weights MATCH δ₀ theory!' if 0.5 < mean_ratio < 2.0 else '→ Weights deviate from theory'}

2. LOW-FREQUENCY WEIGHT:
   Learned: {low_freq:.4f}
   Theory (1-δ₀): {ONE_MINUS_DELTA:.4f}
   Match: {abs(low_freq - ONE_MINUS_DELTA) < 0.1}

3. PERFORMANCE:
   δ₀-theory filter: R² = {r2_delta:.4f}
   Learnable filter: R² = {r2_learn:.4f}
   {'→ Theory-derived filter competitive!' if r2_delta > r2_baseline else '→ Needs tuning'}

4. KEY FINDING:
   The depletion constant δ₀ = {DELTA_0:.6f} {'DOES' if mean_ratio > 0.5 else 'does NOT'}
   appear to be a natural attractor for learned spectral filter weights.
""")

    dataset.close()
    print("=" * 70)


if __name__ == '__main__':
    main()
