#!/usr/bin/env python3
"""
Rigorous Statistical Validation of Spectral DAT.

Tests the optimal configuration (anchor=2.5, sigma=3.5) with:
- Multiple random seeds
- Full training epochs
- Bootstrap confidence intervals
- Paired statistical tests
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
import scipy.stats as stats
import math

from dat_ml.core.device import DEVICE
from dat_ml.core.config import ExperimentConfig
from dat_ml.data.netcdf_loader import ERA5RegionalDataset
from dat_ml.data.loader import create_dataloaders

TAU = (1 + math.sqrt(5)) / 2


class OptimalSpectralDAT(nn.Module):
    """Spectral DAT with optimal parameters from tuning."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Optimal parameters from tuning
        self.anchor = 2.5
        self.base_sigma = 3.5
        n_shells = 7
        shell_start = 0

        shell_targets = [self.anchor * (TAU ** n) for n in range(shell_start, shell_start + n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        self.shell_weights = nn.Parameter(torch.ones(n_shells))
        self.sigma_mult = nn.Parameter(torch.ones(n_shells))
        self.low_freq_weight = nn.Parameter(torch.tensor(0.6))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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

        dat_mask = torch.zeros_like(k_mag)
        for i, target in enumerate(self.shell_targets):
            sigma = self.base_sigma + (target * 0.15)
            sigma = sigma * F.softplus(self.sigma_mult[i])
            weight = F.softplus(self.shell_weights[i])
            dat_mask = dat_mask + weight * torch.exp(-((k_mag - target)**2) / (2 * sigma**2))

        low_freq_mask = (k_mag < 10.0).float()
        dat_mask = dat_mask + low_freq_mask * torch.sigmoid(self.low_freq_weight)
        return dat_mask.clamp(0, 1)

    def forward(self, x):
        seq_len = x.shape[-1]

        x_fft = torch.fft.fft(x, dim=-1)
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)

        mask = self.create_mask(seq_len, x.device)
        x_filtered = x_fft_shifted * mask

        x_out = torch.fft.ifft(torch.fft.ifftshift(x_filtered, dim=-1), dim=-1).real

        x_combined = torch.cat([x, x_out], dim=-1)
        return self.mlp(x_combined)


def create_baseline(input_dim, hidden_dim, output_dim):
    """Create baseline MLP with same capacity."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    )


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Full training with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 25
    no_improve = 0

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
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def evaluate(model, test_loader):
    """Get predictions and targets."""
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
    return preds, targets


def compute_metrics(preds, targets):
    """Compute all metrics."""
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    print("=" * 70)
    print("RIGOROUS STATISTICAL VALIDATION")
    print("Spectral DAT (anchor=2.5, σ=3.5) vs Baseline")
    print("=" * 70)

    N_SEEDS = 10  # Number of independent runs

    # Load data
    print("\nLoading data...")
    nc_path = Path('/Users/bryan/Chaos_analogies/quasicrystal-chaos-analogies/phason_brain/era5_z500_2023_2025.nc')

    dataset = ERA5RegionalDataset(
        nc_path, variable='z', sequence_length=30, horizon=7,
        normalize=True, preload_device='mps'
    )

    sample_x, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = 6
    hidden_dim = 128

    print(f"Data: {len(dataset)} samples")
    print(f"Running {N_SEEDS} independent trials...")

    baseline_results = []
    spectral_results = []
    all_baseline_errors = []
    all_spectral_errors = []

    for seed in range(N_SEEDS):
        print(f"\n--- Trial {seed + 1}/{N_SEEDS} (seed={seed}) ---")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create fresh data splits
        exp_config = ExperimentConfig()
        exp_config.data.seed = seed
        exp_config.training.batch_size = 64
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, exp_config.data, exp_config.training
        )

        # Train baseline
        baseline = create_baseline(input_dim, hidden_dim, output_dim)
        baseline = train_model(baseline, train_loader, val_loader, epochs=100)
        baseline_preds, targets = evaluate(baseline, test_loader)
        baseline_metrics = compute_metrics(baseline_preds, targets)
        baseline_results.append(baseline_metrics)
        baseline_errors = np.abs(baseline_preds - targets)
        all_baseline_errors.append(baseline_errors)

        print(f"  Baseline:     R²={baseline_metrics['r2']:.6f}, RMSE={baseline_metrics['rmse']:.6f}")

        # Train spectral DAT
        spectral = OptimalSpectralDAT(input_dim, hidden_dim, output_dim)
        spectral = train_model(spectral, train_loader, val_loader, epochs=100)
        spectral_preds, _ = evaluate(spectral, test_loader)
        spectral_metrics = compute_metrics(spectral_preds, targets)
        spectral_results.append(spectral_metrics)
        spectral_errors = np.abs(spectral_preds - targets)
        all_spectral_errors.append(spectral_errors)

        print(f"  Spectral DAT: R²={spectral_metrics['r2']:.6f}, RMSE={spectral_metrics['rmse']:.6f}")

        improvement = spectral_metrics['r2'] - baseline_metrics['r2']
        print(f"  Δ R²: {improvement:+.6f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    baseline_r2s = [r['r2'] for r in baseline_results]
    spectral_r2s = [r['r2'] for r in spectral_results]
    baseline_rmses = [r['rmse'] for r in baseline_results]
    spectral_rmses = [r['rmse'] for r in spectral_results]

    print(f"\nBaseline MLP:")
    print(f"  R² = {np.mean(baseline_r2s):.6f} ± {np.std(baseline_r2s):.6f}")
    print(f"  RMSE = {np.mean(baseline_rmses):.6f} ± {np.std(baseline_rmses):.6f}")

    print(f"\nSpectral DAT (anchor=2.5, σ=3.5):")
    print(f"  R² = {np.mean(spectral_r2s):.6f} ± {np.std(spectral_r2s):.6f}")
    print(f"  RMSE = {np.mean(spectral_rmses):.6f} ± {np.std(spectral_rmses):.6f}")

    # Improvement statistics
    improvements = [s - b for s, b in zip(spectral_r2s, baseline_r2s)]
    mean_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)

    print(f"\nR² Improvement:")
    print(f"  Mean: {mean_improvement:+.6f}")
    print(f"  Std:  {std_improvement:.6f}")

    # Error reduction
    baseline_unexplained = 1 - np.mean(baseline_r2s)
    spectral_unexplained = 1 - np.mean(spectral_r2s)
    error_reduction = (baseline_unexplained - spectral_unexplained) / baseline_unexplained * 100
    print(f"  Error reduction: {error_reduction:.1f}%")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Paired t-test on R² values
    t_stat, p_value = stats.ttest_rel(spectral_r2s, baseline_r2s)
    print(f"\nPaired t-test (R² values):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pvalue = stats.wilcoxon(spectral_r2s, baseline_r2s, alternative='greater')
    print(f"\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat:.4f}")
    print(f"  p-value: {w_pvalue:.6f}")

    # Bootstrap CI on improvement
    improvement_ci = bootstrap_ci(improvements, n_bootstrap=10000)
    print(f"\nBootstrap 95% CI on R² improvement:")
    print(f"  [{improvement_ci[0]:+.6f}, {improvement_ci[1]:+.6f}]")

    # Pooled error comparison
    all_baseline = np.concatenate(all_baseline_errors)
    all_spectral = np.concatenate(all_spectral_errors)

    # Use same indices for pairing
    min_len = min(len(all_baseline), len(all_spectral))
    t_error, p_error = stats.ttest_rel(all_spectral[:min_len], all_baseline[:min_len])
    print(f"\nPaired t-test (pooled prediction errors):")
    print(f"  t-statistic: {t_error:.4f}")
    print(f"  p-value: {p_error:.6f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(spectral_r2s) + np.var(baseline_r2s)) / 2)
    cohens_d = mean_improvement / pooled_std if pooled_std > 0 else 0
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if p_value < 0.05 and mean_improvement > 0:
        print(f"\n✓ SIGNIFICANT IMPROVEMENT (p={p_value:.4f})")
        print(f"  Spectral DAT improves R² by {mean_improvement:+.6f}")
        print(f"  This reduces unexplained variance by {error_reduction:.1f}%")
        print(f"  95% CI: [{improvement_ci[0]:+.6f}, {improvement_ci[1]:+.6f}]")

        if cohens_d > 0.8:
            print(f"  Effect size: LARGE (d={cohens_d:.2f})")
        elif cohens_d > 0.5:
            print(f"  Effect size: MEDIUM (d={cohens_d:.2f})")
        elif cohens_d > 0.2:
            print(f"  Effect size: SMALL (d={cohens_d:.2f})")
        else:
            print(f"  Effect size: NEGLIGIBLE (d={cohens_d:.2f})")

    elif p_value < 0.1 and mean_improvement > 0:
        print(f"\n~ MARGINALLY SIGNIFICANT (p={p_value:.4f})")
        print(f"  Trend toward improvement: {mean_improvement:+.6f}")
        print(f"  More data may be needed for conclusive results")

    else:
        print(f"\n✗ NO SIGNIFICANT IMPROVEMENT (p={p_value:.4f})")
        print(f"  Mean difference: {mean_improvement:+.6f}")

    # Win rate
    wins = sum(1 for s, b in zip(spectral_r2s, baseline_r2s) if s > b)
    print(f"\nWin rate: {wins}/{N_SEEDS} trials ({100*wins/N_SEEDS:.0f}%)")

    dataset.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
