#!/usr/bin/env python3
"""
Realistic Forecast Comparison.

Tests spectral DAT against proper forecast baselines:
- Persistence (predict today's value for tomorrow)
- Climatology (predict long-term average)
- Linear regression

Reports results in physical units (meters for Z500).
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from pathlib import Path
import math

from dat_ml.core.device import DEVICE

TAU = (1 + math.sqrt(5)) / 2


class SpectralDATModel(nn.Module):
    """Optimal spectral DAT model."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.anchor = 2.5
        self.base_sigma = 3.5
        n_shells = 7

        shell_targets = [self.anchor * (TAU ** n) for n in range(7)]
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


def load_era5_raw(nc_path, sequence_length=30, horizon=7):
    """Load ERA5 data with regional means, keeping track of normalization."""
    ds = xr.open_dataset(nc_path)
    data_raw = ds['z']

    if 'pressure_level' in data_raw.dims:
        data_raw = data_raw.isel(pressure_level=0)

    lats = ds.latitude.values
    lons = ds.longitude.values

    # Regions
    regions = {
        'north_atlantic': {'lat': (30, 60), 'lon': (280, 360)},
        'north_pacific': {'lat': (30, 60), 'lon': (150, 220)},
        'europe': {'lat': (35, 70), 'lon': (0, 40)},
        'asia': {'lat': (30, 60), 'lon': (80, 140)},
        'arctic': {'lat': (70, 90), 'lon': (0, 360)},
        'tropics': {'lat': (-20, 20), 'lon': (0, 360)},
    }

    regional_means = []
    for name, bounds in regions.items():
        lat_min, lat_max = bounds['lat']
        lon_min, lon_max = bounds['lon']

        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if lon_min < lon_max:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
        else:
            lon_mask = (lons >= lon_min) | (lons <= lon_max)

        weights = np.cos(np.radians(lats[lat_mask]))
        weights = weights / weights.sum()

        region_data = data_raw.values[:, lat_mask, :][:, :, lon_mask]
        region_mean = np.average(region_data.mean(axis=2), axis=1, weights=weights)
        regional_means.append(region_mean)

    data = np.stack(regional_means, axis=1).astype(np.float32)

    # Keep raw stats for denormalization
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)

    # Create samples
    n_samples = len(data) - sequence_length - horizon + 1

    # Split: 70% train, 15% val, 15% test
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    ds.close()

    return {
        'data': data,
        'mean': mean,
        'std': std,
        'n_samples': n_samples,
        'train_end': train_end,
        'val_end': val_end,
        'sequence_length': sequence_length,
        'horizon': horizon,
        'region_names': list(regions.keys())
    }


def persistence_forecast(data_info):
    """Persistence baseline: predict last known value."""
    data = data_info['data']
    seq_len = data_info['sequence_length']
    horizon = data_info['horizon']
    val_end = data_info['val_end']
    n_samples = data_info['n_samples']

    # Test set predictions
    preds = []
    targets = []

    for idx in range(val_end, n_samples):
        # Last value of input sequence
        last_val = data[idx + seq_len - 1]
        # Target value
        target_val = data[idx + seq_len + horizon - 1]

        preds.append(last_val)
        targets.append(target_val)

    return np.array(preds), np.array(targets)


def climatology_forecast(data_info):
    """Climatology baseline: predict training mean."""
    data = data_info['data']
    seq_len = data_info['sequence_length']
    horizon = data_info['horizon']
    train_end = data_info['train_end']
    val_end = data_info['val_end']
    n_samples = data_info['n_samples']

    # Compute climatology from training data
    train_data = data[:train_end + seq_len]
    climatology = train_data.mean(axis=0)

    # Test set predictions
    preds = []
    targets = []

    for idx in range(val_end, n_samples):
        target_val = data[idx + seq_len + horizon - 1]
        preds.append(climatology)
        targets.append(target_val)

    return np.array(preds), np.array(targets)


def train_and_predict(model_class, data_info, hidden_dim=128, epochs=100):
    """Train model and get test predictions."""
    data = data_info['data']
    mean = data_info['mean']
    std = data_info['std']
    seq_len = data_info['sequence_length']
    horizon = data_info['horizon']
    train_end = data_info['train_end']
    val_end = data_info['val_end']
    n_samples = data_info['n_samples']

    # Normalize
    data_norm = (data - mean) / (std + 1e-8)

    # Prepare data
    def get_batch(indices):
        X, Y = [], []
        for idx in indices:
            x = data_norm[idx:idx + seq_len].flatten()
            y = data_norm[idx + seq_len + horizon - 1]
            X.append(x)
            Y.append(y)
        return torch.tensor(np.array(X), device=DEVICE), torch.tensor(np.array(Y), device=DEVICE)

    input_dim = seq_len * data.shape[1]
    output_dim = data.shape[1]

    model = model_class(input_dim, hidden_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))

    best_val = float('inf')
    best_state = None
    patience = 20
    no_improve = 0

    batch_size = 64

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_indices)

        for i in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[i:i+batch_size]
            X, Y = get_batch(batch_idx)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            X_val, Y_val = get_batch(val_indices)
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

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

    # Test predictions
    test_indices = list(range(val_end, n_samples))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for idx in test_indices:
            X = torch.tensor(data_norm[idx:idx + seq_len].flatten(), device=DEVICE).unsqueeze(0)
            pred_norm = model(X).cpu().numpy()[0]

            # Denormalize
            pred = pred_norm * std.flatten() + mean.flatten()
            target = data[idx + seq_len + horizon - 1]

            preds.append(pred)
            targets.append(target)

    return np.array(preds), np.array(targets)


class BaselineMLP(nn.Module):
    """Simple MLP baseline."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
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

    def forward(self, x):
        return self.mlp(x)


def compute_skill_scores(preds, targets, climatology_preds):
    """Compute forecast skill scores."""
    # RMSE in physical units (meters for Z500)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    # MAE
    mae = np.mean(np.abs(preds - targets))

    # Anomaly Correlation Coefficient (ACC)
    # ACC measures correlation of anomalies from climatology
    clim_mean = climatology_preds.mean(axis=0)
    pred_anom = preds - clim_mean
    target_anom = targets - clim_mean

    acc_per_region = []
    for i in range(preds.shape[1]):
        corr = np.corrcoef(pred_anom[:, i], target_anom[:, i])[0, 1]
        acc_per_region.append(corr)

    acc = np.mean(acc_per_region)

    # Skill Score relative to climatology
    mse = np.mean((preds - targets) ** 2)
    mse_clim = np.mean((climatology_preds - targets) ** 2)
    skill_score = 1 - mse / mse_clim

    return {
        'rmse': rmse,
        'mae': mae,
        'acc': acc,
        'skill_score': skill_score
    }


def main():
    print("=" * 70)
    print("REALISTIC FORECAST COMPARISON")
    print("Z500 7-day Regional Forecast")
    print("=" * 70)

    nc_path = Path('/Users/bryan/Chaos_analogies/quasicrystal-chaos-analogies/phason_brain/era5_z500_2023_2025.nc')

    print("\nLoading ERA5 Z500 data...")
    data_info = load_era5_raw(nc_path, sequence_length=30, horizon=7)

    print(f"Total timesteps: {len(data_info['data'])}")
    print(f"Regions: {data_info['region_names']}")
    print(f"Forecast horizon: {data_info['horizon']} days")
    print(f"Test samples: {data_info['n_samples'] - data_info['val_end']}")

    # Z500 typical values
    z500_mean = data_info['mean'].mean()
    z500_std = data_info['std'].mean()
    print(f"\nZ500 statistics:")
    print(f"  Mean: {z500_mean:.0f} m")
    print(f"  Std:  {z500_std:.0f} m")

    results = {}

    # 1. Persistence baseline
    print("\n" + "-" * 50)
    print("PERSISTENCE BASELINE (predict last known value)")
    print("-" * 50)
    pers_preds, pers_targets = persistence_forecast(data_info)
    print(f"  Predicting: tomorrow = today")

    # 2. Climatology baseline
    print("\n" + "-" * 50)
    print("CLIMATOLOGY BASELINE (predict training mean)")
    print("-" * 50)
    clim_preds, clim_targets = climatology_forecast(data_info)
    print(f"  Predicting: constant = training average")

    # 3. MLP baseline
    print("\n" + "-" * 50)
    print("MLP BASELINE")
    print("-" * 50)
    mlp_preds, mlp_targets = train_and_predict(BaselineMLP, data_info)
    print("  Training complete")

    # 4. Spectral DAT
    print("\n" + "-" * 50)
    print("SPECTRAL DAT (anchor=2.5, σ=3.5)")
    print("-" * 50)
    dat_preds, dat_targets = train_and_predict(SpectralDATModel, data_info)
    print("  Training complete")

    # Compute all metrics
    print("\n" + "=" * 70)
    print("RESULTS (Physical Units)")
    print("=" * 70)

    models = {
        'Persistence': (pers_preds, pers_targets),
        'Climatology': (clim_preds, clim_targets),
        'MLP': (mlp_preds, mlp_targets),
        'Spectral DAT': (dat_preds, dat_targets),
    }

    print(f"\n{'Model':<20} {'RMSE (m)':>12} {'MAE (m)':>12} {'ACC':>8} {'Skill':>8}")
    print("-" * 70)

    all_results = {}
    for name, (preds, targets) in models.items():
        metrics = compute_skill_scores(preds, targets, clim_preds)
        all_results[name] = metrics
        print(f"{name:<20} {metrics['rmse']:>12.1f} {metrics['mae']:>12.1f} "
              f"{metrics['acc']:>8.4f} {metrics['skill_score']:>8.4f}")

    # Improvement analysis
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    mlp_rmse = all_results['MLP']['rmse']
    dat_rmse = all_results['Spectral DAT']['rmse']
    pers_rmse = all_results['Persistence']['rmse']

    print(f"\nSpectral DAT vs MLP:")
    print(f"  RMSE reduction: {mlp_rmse - dat_rmse:.1f} meters")
    print(f"  Relative improvement: {(mlp_rmse - dat_rmse) / mlp_rmse * 100:.1f}%")

    print(f"\nSpectral DAT vs Persistence:")
    print(f"  RMSE reduction: {pers_rmse - dat_rmse:.1f} meters")
    print(f"  Relative improvement: {(pers_rmse - dat_rmse) / pers_rmse * 100:.1f}%")

    # Context
    print("\n" + "=" * 70)
    print("CONTEXT: What these numbers mean")
    print("=" * 70)

    print(f"""
Z500 (500 hPa geopotential height) is a key variable for mid-latitude weather.
Typical values: ~5500m, with variations of ~100-300m driving weather patterns.

For 7-day forecasts:
  - Persistence RMSE ~{pers_rmse:.0f}m is the 'do nothing' baseline
  - Operational models (ECMWF) achieve ~50-80m RMSE at day 7
  - Our Spectral DAT: {dat_rmse:.0f}m RMSE

ACC (Anomaly Correlation Coefficient):
  - ACC > 0.6 is considered 'useful' forecast skill
  - ACC > 0.8 is 'good' skill
  - Our Spectral DAT ACC: {all_results['Spectral DAT']['acc']:.3f}
""")

    # The gap
    ecmwf_approx = 70  # Approximate ECMWF 7-day RMSE
    print(f"Gap to operational forecasts: ~{dat_rmse - ecmwf_approx:.0f}m")
    print(f"  (Operational models use: full global grid, multiple variables,")
    print(f"   physics constraints, ensemble methods, decades of development)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
