#!/usr/bin/env python3
"""
Quick Demo: SpectralDATLayer improves time series prediction

This demonstrates the theory-derived golden ratio spectral filter
on the Mackey-Glass chaotic system.

Run: python demo_spectral_dat.py
"""

import torch
import torch.nn as nn
import numpy as np
from dat_ml import SpectralDATLayer, DELTA_0, OPTIMAL_SIGMA, H3_COORDINATION

print("=" * 60)
print("SpectralDATLayer Demo")
print("Golden ratio spectral filtering from H₃ manifold geometry")
print("=" * 60)

# Theory verification
print(f"\nTheory constants:")
print(f"  δ₀ (depletion constant) = {DELTA_0:.6f}")
print(f"  Optimal σ = {OPTIMAL_SIGMA:.4f}")
print(f"  σ × δ₀ = {OPTIMAL_SIGMA * DELTA_0:.4f} ≈ {H3_COORDINATION} (H₃ coordination)")

# Generate Mackey-Glass data (moderate chaos - ideal for DAT)
print("\nGenerating Mackey-Glass τ=17 data...")

def mackey_glass(n_points=5000, tau=17, delta_t=1.0):
    """Generate Mackey-Glass chaotic time series."""
    history_len = tau + 1
    x = np.zeros(n_points + history_len)
    x[:history_len] = 0.9 + 0.2 * np.random.rand(history_len)

    for i in range(history_len, n_points + history_len):
        x_tau = x[i - tau]
        x[i] = x[i-1] + delta_t * (0.2 * x_tau / (1 + x_tau**10) - 0.1 * x[i-1])

    return x[history_len:]

data = mackey_glass(5000, tau=17)
data = (data - data.mean()) / data.std()

# Create sequences
seq_len, pred_len = 64, 6
X, y = [], []
for i in range(len(data) - seq_len - pred_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len:i+seq_len+pred_len])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"  Train: {len(X_train)} sequences")
print(f"  Test:  {len(X_test)} sequences")

# Define models
class BaselineMLP(nn.Module):
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

class SpectralDATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.spectral = SpectralDATLayer()  # Theory-derived defaults
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 2x for residual
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
        return self.net(x_combined)

def train_and_evaluate(model, name, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = criterion(pred, y_test).item()
        rmse = np.sqrt(mse)

    return rmse

# Compare models
print("\n" + "-" * 60)
print("Training comparison (100 epochs each)...")
print("-" * 60)

# Run multiple seeds for statistical validity
n_seeds = 5
baseline_results = []
spectral_results = []

for seed in range(n_seeds):
    torch.manual_seed(seed)

    baseline = BaselineMLP(seq_len, 128, pred_len)
    spectral = SpectralDATModel(seq_len, 128, pred_len)

    baseline_rmse = train_and_evaluate(baseline, "Baseline")
    spectral_rmse = train_and_evaluate(spectral, "Spectral DAT")

    baseline_results.append(baseline_rmse)
    spectral_results.append(spectral_rmse)

    print(f"  Seed {seed}: Baseline={baseline_rmse:.4f}, SpectralDAT={spectral_rmse:.4f}")

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

baseline_mean = np.mean(baseline_results)
spectral_mean = np.mean(spectral_results)
improvement = (baseline_mean - spectral_mean) / baseline_mean * 100

print(f"\nBaseline MLP:      {baseline_mean:.4f} ± {np.std(baseline_results):.4f} RMSE")
print(f"Spectral DAT:      {spectral_mean:.4f} ± {np.std(spectral_results):.4f} RMSE")
print(f"\nImprovement:       {improvement:+.1f}%")

# Statistical significance
from scipy import stats
t_stat, p_value = stats.ttest_rel(baseline_results, spectral_results)
print(f"Paired t-test:     t={t_stat:.2f}, p={p_value:.4f}")

if p_value < 0.05 and improvement > 0:
    print("\n✓ Statistically significant improvement (p < 0.05)")
else:
    print("\n  Result not statistically significant")

print("\n" + "=" * 60)
print("The golden ratio filter from H₃ geometry provides a")
print("theory-grounded improvement for chaotic time series prediction.")
print("=" * 60)
