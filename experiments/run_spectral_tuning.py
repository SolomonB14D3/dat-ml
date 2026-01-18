#!/usr/bin/env python3
"""
Spectral DAT Parameter Tuning.

Since even small improvements matter when baseline is strong,
let's find the optimal golden ratio anchor and shell configuration.
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
from dat_ml.core.config import ExperimentConfig
from dat_ml.data.netcdf_loader import ERA5RegionalDataset
from dat_ml.data.loader import create_dataloaders

TAU = (1 + math.sqrt(5)) / 2


class TunableSpectralDAT(nn.Module):
    """Spectral DAT with configurable parameters."""

    def __init__(self, input_dim, hidden_dim, output_dim, anchor, n_shells, shell_start, base_sigma):
        super().__init__()

        self.anchor = anchor
        self.base_sigma = base_sigma

        # Shell targets
        shell_targets = [anchor * (TAU ** n) for n in range(shell_start, shell_start + n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        # Learnable parameters
        self.shell_weights = nn.Parameter(torch.ones(n_shells))
        self.sigma_mult = nn.Parameter(torch.ones(n_shells))
        self.low_freq_weight = nn.Parameter(torch.tensor(0.6))

        # MLP
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


def quick_train(model, train_loader, val_loader, epochs=50):
    """Quick training for parameter search."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
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
        best_val = min(best_val, val_loss)

    return best_val


def evaluate(model, test_loader):
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
    print("=" * 70)
    print("SPECTRAL DAT PARAMETER TUNING")
    print("Finding optimal golden ratio configuration")
    print("=" * 70)

    # Load data
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

    sample_x, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = 6
    hidden_dim = 128

    # First, get baseline
    print("\nTraining baseline...")
    baseline = nn.Sequential(
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
    ).to(DEVICE)

    quick_train(baseline, train_loader, val_loader, epochs=80)
    baseline_r2 = evaluate(baseline, test_loader)
    baseline_error = 1 - baseline_r2
    print(f"Baseline R²: {baseline_r2:.6f} (unexplained: {baseline_error*100:.4f}%)")

    # Parameter grid
    anchors = [2.5, 5.0, 10.0, 15.0, 21.0, 30.0]  # Your weather scripts used 2.5 and 21.0
    n_shells_list = [5, 7, 9, 11]
    shell_starts = [-3, -2, -1, 0]
    sigmas = [3.5, 10.0, 15.0]

    print(f"\nSearching {len(anchors) * len(n_shells_list) * len(shell_starts) * len(sigmas)} configurations...")

    results = []
    best_r2 = baseline_r2
    best_config = None

    total = len(anchors) * len(n_shells_list) * len(shell_starts) * len(sigmas)
    count = 0

    for anchor in anchors:
        for n_shells in n_shells_list:
            for shell_start in shell_starts:
                for sigma in sigmas:
                    count += 1

                    model = TunableSpectralDAT(
                        input_dim, hidden_dim, output_dim,
                        anchor=anchor, n_shells=n_shells,
                        shell_start=shell_start, base_sigma=sigma
                    )

                    quick_train(model, train_loader, val_loader, epochs=50)
                    r2 = evaluate(model, test_loader)

                    error_reduction = ((baseline_error - (1 - r2)) / baseline_error) * 100

                    config = {
                        'anchor': anchor, 'n_shells': n_shells,
                        'shell_start': shell_start, 'sigma': sigma,
                        'r2': r2, 'error_reduction': error_reduction
                    }
                    results.append(config)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_config = config
                        print(f"[{count}/{total}] NEW BEST: anchor={anchor}, shells={n_shells}, "
                              f"start={shell_start}, σ={sigma} → R²={r2:.6f} "
                              f"(error reduced by {error_reduction:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)

    results.sort(key=lambda x: -x['r2'])

    print(f"{'Anchor':>8} {'Shells':>8} {'Start':>8} {'Sigma':>8} {'R²':>10} {'Error Δ':>10}")
    print("-" * 70)

    for r in results[:10]:
        print(f"{r['anchor']:>8.1f} {r['n_shells']:>8} {r['shell_start']:>8} "
              f"{r['sigma']:>8.1f} {r['r2']:>10.6f} {r['error_reduction']:>+9.1f}%")

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    if best_config:
        print(f"Anchor: {best_config['anchor']}")
        print(f"N shells: {best_config['n_shells']}")
        print(f"Shell start: {best_config['shell_start']}")
        print(f"Base sigma: {best_config['sigma']}")
        print(f"R²: {best_config['r2']:.6f}")
        print(f"Baseline R²: {baseline_r2:.6f}")
        print(f"Improvement: {best_config['r2'] - baseline_r2:.6f}")
        print(f"Error reduction: {best_config['error_reduction']:.1f}%")

    dataset.close()


if __name__ == '__main__':
    main()
