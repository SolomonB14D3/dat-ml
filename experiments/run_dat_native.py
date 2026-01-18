#!/usr/bin/env python3
"""
DAT Native Data Experiment.

Test spectral DAT on data that was GENERATED from DAT principles.
This should be the "home turf" - if golden ratio structure helps anywhere,
it should help here.

Data: dat_gpu_discovery.json, dat_infinite_scaling.json
- frame: time index
- pos: 3D position
- vorticity: scalar vorticity
- offset_6d: 6D offset in E6 lattice space
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import math
from collections import defaultdict

from dat_ml.core.device import DEVICE

TAU = (1 + math.sqrt(5)) / 2


def load_dat_data(json_paths):
    """Load DAT simulation data from JSON files."""
    all_data = []

    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
            all_data.extend(data)

    # Group by frame
    frames = defaultdict(list)
    for entry in all_data:
        frames[entry['frame']].append(entry)

    # Sort frames
    frame_nums = sorted(frames.keys())

    print(f"Loaded {len(all_data)} entries across {len(frame_nums)} frames")

    return frames, frame_nums


def create_timeseries_dataset(frames, frame_nums, seq_len=20, horizon=5):
    """
    Create time series dataset from DAT frames.

    For each frame, we have multiple particles with (pos, vorticity, offset_6d).
    Task: Given sequence of frame statistics, predict future vorticity.
    """
    # Compute per-frame statistics
    frame_stats = []

    for frame_num in frame_nums:
        entries = frames[frame_num]

        # Extract arrays
        positions = np.array([e['pos'] for e in entries])
        vorticities = np.array([e['vorticity'] for e in entries])
        offsets = np.array([e['offset_6d'] for e in entries])

        # Statistics
        stats = np.concatenate([
            # Position stats
            positions.mean(axis=0),  # 3
            positions.std(axis=0),   # 3
            # Vorticity stats
            [vorticities.mean(), vorticities.std(), vorticities.min(), vorticities.max()],  # 4
            # Offset stats (6D)
            offsets.mean(axis=0),    # 6
            offsets.std(axis=0),     # 6
        ])

        frame_stats.append(stats)

    frame_stats = np.array(frame_stats, dtype=np.float32)
    print(f"Frame statistics shape: {frame_stats.shape}")

    # Create sequences
    X, y = [], []
    for i in range(len(frame_stats) - seq_len - horizon + 1):
        x_seq = frame_stats[i:i + seq_len].flatten()
        # Target: predict next frame's vorticity stats
        y_target = frame_stats[i + seq_len + horizon - 1, 6:10]  # vorticity mean, std, min, max

        X.append(x_seq)
        y.append(y_target)

    return np.array(X), np.array(y), frame_stats.shape[1]


def create_particle_dataset(frames, frame_nums, history=5):
    """
    Particle-level prediction: given a particle's history, predict future state.
    """
    # Track particles across frames (by position similarity)
    # Simplified: use first N particles that appear consistently

    # Get consistent particle count
    min_particles = min(len(frames[f]) for f in frame_nums)
    print(f"Using {min_particles} particles per frame")

    # Build trajectories (assuming particle order is consistent)
    X, y = [], []

    for p_idx in range(min_particles):
        # Get this particle's trajectory
        trajectory = []
        for frame_num in frame_nums:
            entry = frames[frame_num][p_idx]
            state = np.concatenate([
                entry['pos'],
                [entry['vorticity']],
                entry['offset_6d']
            ])
            trajectory.append(state)

        trajectory = np.array(trajectory)

        # Create sequences
        for t in range(len(trajectory) - history - 1):
            x_seq = trajectory[t:t + history].flatten()
            y_target = trajectory[t + history]  # Next state (10D)

            X.append(x_seq)
            y.append(y_target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SpectralDATModel(nn.Module):
    """Spectral DAT with optimal parameters."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.anchor = 2.5
        self.base_sigma = 3.5
        n_shells = 7

        shell_targets = [self.anchor * (TAU ** n) for n in range(n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        self.shell_weights = nn.Parameter(torch.ones(n_shells))
        self.sigma_mult = nn.Parameter(torch.ones(n_shells))
        self.low_freq_weight = nn.Parameter(torch.tensor(0.6))

        self.net = nn.Sequential(
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
        return self.net(x_combined)


class BaselineModel(nn.Module):
    """MLP baseline."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(x)


def train_and_evaluate(model_class, X, y, hidden_dim=128, epochs=100, n_runs=5):
    """Train and evaluate with multiple runs."""
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    # Normalize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    results = []

    for run in range(n_runs):
        # Random split
        np.random.seed(run)
        indices = np.random.permutation(len(X))

        train_end = int(len(X) * 0.7)
        val_end = int(len(X) * 0.85)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        X_train, y_train = X_norm[train_idx], y_norm[train_idx]
        X_val, y_val = X_norm[val_idx], y_norm[val_idx]
        X_test, y_test = X_norm[test_idx], y_norm[test_idx]

        # Convert to tensors
        X_train_t = torch.tensor(X_train, device=DEVICE)
        y_train_t = torch.tensor(y_train, device=DEVICE)
        X_val_t = torch.tensor(X_val, device=DEVICE)
        y_val_t = torch.tensor(y_val, device=DEVICE)
        X_test_t = torch.tensor(X_test, device=DEVICE)
        y_test_t = torch.tensor(y_test, device=DEVICE)

        # Train
        model = model_class(input_dim, hidden_dim, output_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        batch_size = 64

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train_t))

            for i in range(0, len(perm), batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                pred = model(X_train_t[idx])
                loss = criterion(pred, y_train_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
            model.to(DEVICE)

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()
            target = y_test_t.cpu().numpy()

        # Denormalize for interpretable metrics
        pred_denorm = pred * y_std + y_mean
        target_denorm = target * y_std + y_mean

        # Metrics
        mse = np.mean((pred_denorm - target_denorm) ** 2)
        mae = np.mean(np.abs(pred_denorm - target_denorm))

        # R² per output
        r2_scores = []
        for i in range(output_dim):
            ss_res = np.sum((target_denorm[:, i] - pred_denorm[:, i]) ** 2)
            ss_tot = np.sum((target_denorm[:, i] - target_denorm[:, i].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_scores.append(r2)

        results.append({
            'mse': mse,
            'mae': mae,
            'r2_mean': np.mean(r2_scores),
            'r2_per_target': r2_scores
        })

    return results


def main():
    print("=" * 70)
    print("DAT NATIVE DATA EXPERIMENT")
    print("Testing spectral DAT on data generated from DAT principles")
    print("=" * 70)

    # Load data
    json_paths = [
        Path('/Users/bryan/dat_gpu_discovery.json'),
        Path('/Users/bryan/dat_infinite_scaling.json')
    ]

    frames, frame_nums = load_dat_data(json_paths)

    hidden_dim = 128
    n_runs = 5

    # =========================================================
    # EXPERIMENT 1: Frame-level prediction
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: FRAME STATISTICS PREDICTION")
    print("Predict future vorticity statistics from past frames")
    print("=" * 60)

    X, y, _ = create_timeseries_dataset(frames, frame_nums, seq_len=20, horizon=5)
    print(f"Dataset: {len(X)} samples, input={X.shape[1]}, output={y.shape[1]}")

    print("\nTraining Baseline MLP...")
    baseline_results = train_and_evaluate(BaselineModel, X, y, hidden_dim, epochs=100, n_runs=n_runs)

    print("Training Spectral DAT...")
    spectral_results = train_and_evaluate(SpectralDATModel, X, y, hidden_dim, epochs=100, n_runs=n_runs)

    print(f"\n{'Model':<20} {'MSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 60)

    baseline_r2 = np.mean([r['r2_mean'] for r in baseline_results])
    baseline_r2_std = np.std([r['r2_mean'] for r in baseline_results])
    print(f"{'Baseline MLP':<20} {np.mean([r['mse'] for r in baseline_results]):>12.4f} "
          f"{np.mean([r['mae'] for r in baseline_results]):>12.4f} "
          f"{baseline_r2:>12.4f} ± {baseline_r2_std:.4f}")

    spectral_r2 = np.mean([r['r2_mean'] for r in spectral_results])
    spectral_r2_std = np.std([r['r2_mean'] for r in spectral_results])
    print(f"{'Spectral DAT':<20} {np.mean([r['mse'] for r in spectral_results]):>12.4f} "
          f"{np.mean([r['mae'] for r in spectral_results]):>12.4f} "
          f"{spectral_r2:>12.4f} ± {spectral_r2_std:.4f}")

    improvement = spectral_r2 - baseline_r2
    print(f"\nR² Improvement: {improvement:+.4f}")

    # =========================================================
    # EXPERIMENT 2: Particle trajectory prediction
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: PARTICLE TRAJECTORY PREDICTION")
    print("Predict next particle state (pos, vorticity, offset_6d)")
    print("=" * 60)

    X, y = create_particle_dataset(frames, frame_nums, history=10)
    print(f"Dataset: {len(X)} samples, input={X.shape[1]}, output={y.shape[1]}")

    print("\nTraining Baseline MLP...")
    baseline_results = train_and_evaluate(BaselineModel, X, y, hidden_dim, epochs=100, n_runs=n_runs)

    print("Training Spectral DAT...")
    spectral_results = train_and_evaluate(SpectralDATModel, X, y, hidden_dim, epochs=100, n_runs=n_runs)

    print(f"\n{'Model':<20} {'MSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 60)

    baseline_r2 = np.mean([r['r2_mean'] for r in baseline_results])
    baseline_r2_std = np.std([r['r2_mean'] for r in baseline_results])
    print(f"{'Baseline MLP':<20} {np.mean([r['mse'] for r in baseline_results]):>12.4f} "
          f"{np.mean([r['mae'] for r in baseline_results]):>12.4f} "
          f"{baseline_r2:>12.4f} ± {baseline_r2_std:.4f}")

    spectral_r2 = np.mean([r['r2_mean'] for r in spectral_results])
    spectral_r2_std = np.std([r['r2_mean'] for r in spectral_results])
    print(f"{'Spectral DAT':<20} {np.mean([r['mse'] for r in spectral_results]):>12.4f} "
          f"{np.mean([r['mae'] for r in spectral_results]):>12.4f} "
          f"{spectral_r2:>12.4f} ± {spectral_r2_std:.4f}")

    improvement = spectral_r2 - baseline_r2
    print(f"\nR² Improvement: {improvement:+.4f}")

    # Per-component analysis
    print("\nPer-component R² (last run):")
    component_names = ['pos_x', 'pos_y', 'pos_z', 'vorticity',
                       'off_1', 'off_2', 'off_3', 'off_4', 'off_5', 'off_6']

    print(f"{'Component':<12} {'Baseline':>10} {'Spectral':>10} {'Δ':>10}")
    print("-" * 45)

    for i, name in enumerate(component_names):
        b_r2 = baseline_results[-1]['r2_per_target'][i]
        s_r2 = spectral_results[-1]['r2_per_target'][i]
        print(f"{name:<12} {b_r2:>10.4f} {s_r2:>10.4f} {s_r2-b_r2:>+10.4f}")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Does spectral DAT help on DAT-native data?")
    print("=" * 70)


if __name__ == '__main__':
    main()
