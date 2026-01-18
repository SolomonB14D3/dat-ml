#!/usr/bin/env python3
"""
Spectral DAT Experiment.

Tests if applying DAT principles in SPECTRAL space (like your weather
skeleton scripts) works better than the feature-space approach.

Your weather scripts showed that golden ratio frequency filtering
reveals structure - can this help prediction?
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import scipy.stats as stats

from dat_ml.core.device import DEVICE, HARDWARE
from dat_ml.core.config import ExperimentConfig
from dat_ml.data.netcdf_loader import ERA5RegionalDataset
from dat_ml.data.loader import create_dataloaders
from dat_ml.transforms.spectral_dat import SpectralDATPredictor, SpectralDATLayer, SpectralBoostLayer
from dat_ml.models.baseline import StandardMLP


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, name="Model"):
    """Train model with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()

            if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
                out = model(x)
                if isinstance(out, dict):
                    pred = out['prediction']
                else:
                    pred = out
            else:
                pred = model(x)

            loss = criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)

                out = model(x)
                if isinstance(out, dict):
                    pred = out['prediction']
                else:
                    pred = out

                val_loss += criterion(pred.squeeze(), y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if no_improve >= patience:
            print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model, best_val


def evaluate_model(model, test_loader):
    """Evaluate on blind test set."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            out = model(x)
            if isinstance(out, dict):
                pred = out['prediction']
            else:
                pred = out

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(preds, targets)[0, 1]

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr, 'preds': preds, 'targets': targets}


class SpectralMLPHybrid(nn.Module):
    """MLP with spectral DAT preprocessing layer."""

    def __init__(self, input_dim, hidden_dim, output_dim, use_boost=False):
        super().__init__()

        if use_boost:
            self.spectral = SpectralBoostLayer(learnable=True)
        else:
            self.spectral = SpectralDATLayer(learnable=True)

        # Concatenate original + spectrally filtered
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

    def forward(self, x):
        x_filtered = self.spectral(x)
        x_combined = torch.cat([x, x_filtered], dim=-1)
        return self.mlp(x_combined)


def main():
    print("=" * 70)
    print("SPECTRAL DAT EXPERIMENT")
    print("Testing golden-ratio frequency filtering from weather scripts")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Hardware: {HARDWARE.total_cores} cores, {HARDWARE.memory_gb}GB RAM")

    # Load data
    print("\nLoading ERA5 data...")
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

    sample_x, sample_y = dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()
    hidden_dim = 128

    print(f"Data: {len(dataset)} samples, input={input_dim}, output={output_dim}")

    # Models to test
    models = {}

    print("\n" + "=" * 60)
    print("MODEL 1: BASELINE MLP")
    print("=" * 60)
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
    )
    n_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {n_params:,}")
    baseline, _ = train_model(baseline, train_loader, val_loader, name="Baseline")
    models['Baseline MLP'] = (baseline, evaluate_model(baseline, test_loader))

    print("\n" + "=" * 60)
    print("MODEL 2: SPECTRAL DAT (Filter Mode)")
    print("  Applies golden-ratio frequency MASK (like weather scripts)")
    print("=" * 60)
    spectral_filter = SpectralMLPHybrid(input_dim, hidden_dim, output_dim, use_boost=False)
    n_params = sum(p.numel() for p in spectral_filter.parameters())
    print(f"Parameters: {n_params:,}")
    spectral_filter, _ = train_model(spectral_filter, train_loader, val_loader, name="Spectral Filter")
    models['Spectral DAT (Filter)'] = (spectral_filter, evaluate_model(spectral_filter, test_loader))

    print("\n" + "=" * 60)
    print("MODEL 3: SPECTRAL DAT (Boost Mode)")
    print("  BOOSTS golden-ratio frequencies (like boldweather.py)")
    print("=" * 60)
    spectral_boost = SpectralMLPHybrid(input_dim, hidden_dim, output_dim, use_boost=True)
    n_params = sum(p.numel() for p in spectral_boost.parameters())
    print(f"Parameters: {n_params:,}")
    spectral_boost, _ = train_model(spectral_boost, train_loader, val_loader, name="Spectral Boost")
    models['Spectral DAT (Boost)'] = (spectral_boost, evaluate_model(spectral_boost, test_loader))

    print("\n" + "=" * 60)
    print("MODEL 4: SPECTRAL DAT PREDICTOR")
    print("  Full spectral DAT model with residual connection")
    print("=" * 60)
    spectral_pred = SpectralDATPredictor(
        input_dim, hidden_dim, output_dim,
        n_shells=7, anchor=21.0, use_residual=True
    )
    n_params = sum(p.numel() for p in spectral_pred.parameters())
    print(f"Parameters: {n_params:,}")
    spectral_pred, _ = train_model(spectral_pred, train_loader, val_loader, name="Spectral Predictor")
    models['Spectral DAT Predictor'] = (spectral_pred, evaluate_model(spectral_pred, test_loader))

    # Results
    print("\n" + "=" * 70)
    print("BLIND TEST RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Corr':>10}")
    print("-" * 70)

    baseline_r2 = models['Baseline MLP'][1]['r2']
    results_sorted = sorted(models.items(), key=lambda x: -x[1][1]['r2'])

    for name, (model, metrics) in results_sorted:
        delta = metrics['r2'] - baseline_r2
        delta_str = f"({delta:+.4f})" if name != 'Baseline MLP' else ""
        print(f"{name:<30} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f} {metrics['r2']:>10.4f} {metrics['corr']:>10.4f} {delta_str}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    baseline_preds = models['Baseline MLP'][1]['preds']
    targets = models['Baseline MLP'][1]['targets']
    baseline_errors = np.abs(baseline_preds - targets)

    for name, (model, metrics) in models.items():
        if name == 'Baseline MLP':
            continue

        errors = np.abs(metrics['preds'] - targets)
        t_stat, p_val = stats.ttest_rel(errors, baseline_errors)

        print(f"\n{name} vs Baseline:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.6f}")

        if p_val < 0.05:
            if t_stat < 0:
                print(f"  → {name} is SIGNIFICANTLY BETTER (p < 0.05)")
            else:
                print(f"  → Baseline is significantly better (p < 0.05)")
        else:
            print(f"  → No significant difference")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_model = max(models.items(), key=lambda x: x[1][1]['r2'])
    print(f"\nBest performer: {best_model[0]} (R²={best_model[1][1]['r2']:.4f})")

    spectral_models = {k: v for k, v in models.items() if 'Spectral' in k}
    best_spectral = max(spectral_models.items(), key=lambda x: x[1][1]['r2'])

    if best_spectral[1][1]['r2'] > baseline_r2:
        print(f"\n✓ Spectral DAT approach HELPS!")
        print(f"  Best spectral: {best_spectral[0]} (R²={best_spectral[1][1]['r2']:.4f})")
        print(f"  Improvement over baseline: {best_spectral[1][1]['r2'] - baseline_r2:.4f}")
    else:
        print(f"\n✗ Spectral DAT approach does not help on this task")
        print(f"  Best spectral: {best_spectral[0]} (R²={best_spectral[1][1]['r2']:.4f})")
        print(f"  Gap from baseline: {best_spectral[1][1]['r2'] - baseline_r2:.4f}")

    # Cleanup
    dataset.close()

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
