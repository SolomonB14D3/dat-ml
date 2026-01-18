#!/usr/bin/env python3
"""
WOW Signal / SETI Detection Experiment.

The WOW signal (1977) was a 72-second narrowband radio burst at 1420 MHz.
Original data: "6EQUJ5" representing intensity spike from 6→30→30→26→19→5.

This experiment tests if spectral DAT can help:
1. Detect faint signals in noise
2. Predict signal characteristics
3. Identify golden-ratio patterns in signal structure

We generate synthetic SETI-like signals and test detection/prediction.
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
from typing import Tuple

from dat_ml.core.device import DEVICE

TAU = (1 + math.sqrt(5)) / 2  # Golden ratio

# Original WOW signal intensity values (scaled 0-35)
# Characters: space=0, 1-9=1-9, A=10, B=11, ... U=30
WOW_SEQUENCE = [6, 14, 26, 30, 19, 5]  # "6EQUJ5"


def generate_wow_like_signal(
    n_samples: int = 10000,
    signal_length: int = 128,
    noise_level: float = 1.0,
    signal_strength: float = 3.0,
    include_golden_ratio: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic WOW-like signals in noise.

    Returns:
        signals: (n_samples, signal_length) - noisy observations
        labels: (n_samples,) - 1 if signal present, 0 if just noise
        params: (n_samples, 4) - [peak_time, peak_intensity, duration, frequency]
    """
    signals = []
    labels = []
    params = []

    for i in range(n_samples):
        # Base noise (Gaussian)
        noise = np.random.randn(signal_length) * noise_level

        # 50% chance of containing a signal
        has_signal = np.random.random() > 0.5

        if has_signal:
            # Signal parameters
            peak_time = np.random.randint(20, signal_length - 20)
            peak_intensity = signal_strength * (0.5 + np.random.random())
            duration = np.random.randint(5, 20)

            if include_golden_ratio:
                # Golden ratio modulated signal
                freq = TAU * (1 + 0.5 * np.random.randn())
            else:
                freq = 2 + 3 * np.random.random()

            # Create WOW-like envelope (Gaussian rise and fall)
            t = np.arange(signal_length)
            envelope = peak_intensity * np.exp(-((t - peak_time) ** 2) / (2 * (duration/2) ** 2))

            # Narrowband carrier (like hydrogen line)
            carrier = np.sin(2 * np.pi * freq * t / signal_length)

            signal = envelope * carrier
            observation = noise + signal

            params.append([peak_time / signal_length, peak_intensity, duration, freq])
        else:
            observation = noise
            params.append([0, 0, 0, 0])

        signals.append(observation)
        labels.append(1 if has_signal else 0)

    return (
        np.array(signals, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(params, dtype=np.float32)
    )


def generate_golden_ratio_signals(
    n_samples: int = 5000,
    signal_length: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate signals with golden ratio frequency structure.

    If DAT principles are universal, spectral DAT should excel at
    detecting/predicting these signals.
    """
    signals = []
    targets = []

    for i in range(n_samples):
        # Base noise
        noise = np.random.randn(signal_length) * 0.5

        # Golden ratio frequency components
        t = np.arange(signal_length)
        signal = np.zeros(signal_length)

        # Fundamental frequency
        f0 = 2 + np.random.random() * 2

        # Add harmonics at golden ratio intervals
        n_harmonics = np.random.randint(3, 7)
        amplitudes = []
        for h in range(n_harmonics):
            freq = f0 * (TAU ** h)
            amp = 1.0 / (h + 1)  # Decreasing amplitude
            phase = np.random.random() * 2 * np.pi
            signal += amp * np.sin(2 * np.pi * freq * t / signal_length + phase)
            amplitudes.append(amp)

        observation = noise + signal

        # Target: predict the next segment (autoregressive)
        # Or predict the fundamental frequency and n_harmonics
        target = np.array([f0, n_harmonics, np.sum(amplitudes)])

        signals.append(observation)
        targets.append(target)

    return np.array(signals, dtype=np.float32), np.array(targets, dtype=np.float32)


class SpectralDATDetector(nn.Module):
    """Spectral DAT for signal detection."""

    def __init__(self, input_dim, hidden_dim, output_dim, task='classification'):
        super().__init__()
        self.task = task

        # Optimal DAT parameters
        self.anchor = 2.5
        self.base_sigma = 3.5
        n_shells = 7

        shell_targets = [self.anchor * (TAU ** n) for n in range(n_shells)]
        self.register_buffer('shell_targets', torch.tensor(shell_targets))

        self.shell_weights = nn.Parameter(torch.ones(n_shells))
        self.sigma_mult = nn.Parameter(torch.ones(n_shells))
        self.low_freq_weight = nn.Parameter(torch.tensor(0.6))

        # Concatenate original + filtered
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(hidden_dim, output_dim)

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
        features = self.encoder(x_combined)
        return self.head(features)


class BaselineDetector(nn.Module):
    """Simple MLP baseline."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_detector(model, X_train, y_train, X_val, y_val, epochs=100, task='classification'):
    """Train signal detector."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    best_val = float('inf')
    best_state = None
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)

            if task == 'classification':
                loss = criterion(pred.squeeze(), y_batch)
            else:
                loss = criterion(pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if task == 'classification':
                val_loss = criterion(val_pred.squeeze(), y_val_t).item()
            else:
                val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def evaluate_detector(model, X_test, y_test, task='classification'):
    """Evaluate detector performance."""
    model.eval()
    X_test_t = torch.tensor(X_test, device=DEVICE)

    with torch.no_grad():
        pred = model(X_test_t).cpu().numpy()

    if task == 'classification':
        pred_binary = (pred.squeeze() > 0).astype(float)
        accuracy = (pred_binary == y_test).mean()

        # Precision, recall, F1
        tp = ((pred_binary == 1) & (y_test == 1)).sum()
        fp = ((pred_binary == 1) & (y_test == 0)).sum()
        fn = ((pred_binary == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        mse = np.mean((pred - y_test) ** 2)
        mae = np.mean(np.abs(pred - y_test))

        # R² per target
        r2_scores = []
        for i in range(y_test.shape[1]):
            ss_res = np.sum((y_test[:, i] - pred[:, i]) ** 2)
            ss_tot = np.sum((y_test[:, i] - y_test[:, i].mean()) ** 2)
            r2_scores.append(1 - ss_res / ss_tot)

        return {'mse': mse, 'mae': mae, 'r2_mean': np.mean(r2_scores), 'r2_per_target': r2_scores}


def main():
    print("=" * 70)
    print("WOW SIGNAL / SETI DETECTION EXPERIMENT")
    print("=" * 70)
    print(f"\nOriginal WOW signal (1977): '6EQUJ5' = {WOW_SEQUENCE}")
    print("72-second narrowband burst at 1420 MHz (hydrogen line)")

    hidden_dim = 128

    # =========================================================
    # EXPERIMENT 1: Signal Detection (Classification)
    # =========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SIGNAL DETECTION")
    print("Can we detect faint signals in noise?")
    print("=" * 70)

    signal_length = 128

    # Test at different SNR levels
    for snr_label, (noise, signal) in [
        ("Easy (SNR=3)", (1.0, 3.0)),
        ("Medium (SNR=1)", (1.0, 1.0)),
        ("Hard (SNR=0.5)", (1.0, 0.5)),
        ("Very Hard (SNR=0.3)", (1.0, 0.3)),
    ]:
        print(f"\n--- {snr_label} ---")

        # Generate data
        X, y, _ = generate_wow_like_signal(
            n_samples=5000, signal_length=signal_length,
            noise_level=noise, signal_strength=signal
        )

        # Split
        train_end = int(len(X) * 0.7)
        val_end = int(len(X) * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Baseline
        baseline = BaselineDetector(signal_length, hidden_dim, 1)
        baseline = train_detector(baseline, X_train, y_train, X_val, y_val, epochs=50)
        baseline_metrics = evaluate_detector(baseline, X_test, y_test)

        # Spectral DAT
        spectral = SpectralDATDetector(signal_length, hidden_dim, 1)
        spectral = train_detector(spectral, X_train, y_train, X_val, y_val, epochs=50)
        spectral_metrics = evaluate_detector(spectral, X_test, y_test)

        print(f"  Baseline:     Acc={baseline_metrics['accuracy']:.3f}, F1={baseline_metrics['f1']:.3f}")
        print(f"  Spectral DAT: Acc={spectral_metrics['accuracy']:.3f}, F1={spectral_metrics['f1']:.3f}")
        improvement = spectral_metrics['f1'] - baseline_metrics['f1']
        print(f"  F1 Δ: {improvement:+.3f}")

    # =========================================================
    # EXPERIMENT 2: Golden Ratio Signal Prediction
    # =========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: GOLDEN RATIO SIGNAL PREDICTION")
    print("Signals with φ-harmonic structure - DAT's 'home turf'")
    print("=" * 70)

    signal_length = 256
    X, y = generate_golden_ratio_signals(n_samples=5000, signal_length=signal_length)

    # Split
    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\nPredicting: fundamental freq, n_harmonics, total amplitude")

    # Baseline
    baseline = BaselineDetector(signal_length, hidden_dim, 3)
    baseline = train_detector(baseline, X_train, y_train, X_val, y_val, epochs=80, task='regression')
    baseline_metrics = evaluate_detector(baseline, X_test, y_test, task='regression')

    # Spectral DAT
    spectral = SpectralDATDetector(signal_length, hidden_dim, 3, task='regression')
    spectral = train_detector(spectral, X_train, y_train, X_val, y_val, epochs=80, task='regression')
    spectral_metrics = evaluate_detector(spectral, X_test, y_test, task='regression')

    print(f"\nBaseline MLP:")
    print(f"  MSE: {baseline_metrics['mse']:.4f}")
    print(f"  Mean R²: {baseline_metrics['r2_mean']:.4f}")
    print(f"  R² per target: {[f'{r:.3f}' for r in baseline_metrics['r2_per_target']]}")

    print(f"\nSpectral DAT:")
    print(f"  MSE: {spectral_metrics['mse']:.4f}")
    print(f"  Mean R²: {spectral_metrics['r2_mean']:.4f}")
    print(f"  R² per target: {[f'{r:.3f}' for r in spectral_metrics['r2_per_target']]}")

    improvement = spectral_metrics['r2_mean'] - baseline_metrics['r2_mean']
    print(f"\nR² Improvement: {improvement:+.4f}")

    # =========================================================
    # EXPERIMENT 3: Reconstruct WOW-like signal pattern
    # =========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: WOW SIGNAL PATTERN RECONSTRUCTION")
    print("Given noisy observation, predict clean signal parameters")
    print("=" * 70)

    signal_length = 128
    X, y_class, params = generate_wow_like_signal(
        n_samples=8000, signal_length=signal_length,
        noise_level=0.8, signal_strength=2.0
    )

    # Only use samples with signals for parameter prediction
    signal_mask = y_class == 1
    X_signal = X[signal_mask]
    params_signal = params[signal_mask]

    # Split
    n = len(X_signal)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X_signal[:train_end], params_signal[:train_end]
    X_val, y_val = X_signal[train_end:val_end], params_signal[train_end:val_end]
    X_test, y_test = X_signal[val_end:], params_signal[val_end:]

    print(f"\nPredicting: [peak_time, peak_intensity, duration, frequency]")
    print(f"Training samples: {len(X_train)}")

    # Baseline
    baseline = BaselineDetector(signal_length, hidden_dim, 4)
    baseline = train_detector(baseline, X_train, y_train, X_val, y_val, epochs=80, task='regression')
    baseline_metrics = evaluate_detector(baseline, X_test, y_test, task='regression')

    # Spectral DAT
    spectral = SpectralDATDetector(signal_length, hidden_dim, 4, task='regression')
    spectral = train_detector(spectral, X_train, y_train, X_val, y_val, epochs=80, task='regression')
    spectral_metrics = evaluate_detector(spectral, X_test, y_test, task='regression')

    param_names = ['peak_time', 'intensity', 'duration', 'frequency']

    print(f"\n{'Parameter':<12} {'Baseline R²':>12} {'Spectral R²':>12} {'Δ':>8}")
    print("-" * 50)
    for i, name in enumerate(param_names):
        b_r2 = baseline_metrics['r2_per_target'][i]
        s_r2 = spectral_metrics['r2_per_target'][i]
        delta = s_r2 - b_r2
        print(f"{name:<12} {b_r2:>12.4f} {s_r2:>12.4f} {delta:>+8.4f}")

    print(f"\n{'MEAN':<12} {baseline_metrics['r2_mean']:>12.4f} {spectral_metrics['r2_mean']:>12.4f} "
          f"{spectral_metrics['r2_mean'] - baseline_metrics['r2_mean']:>+8.4f}")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The spectral DAT approach applies golden-ratio frequency filtering
to signal detection and parameter estimation.

Key question: Does the φ-based spectral structure help?
""")


if __name__ == '__main__':
    main()
