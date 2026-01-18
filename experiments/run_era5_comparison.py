#!/usr/bin/env python3
"""
ERA5 Z500 Prediction: DAT-H3 vs Baseline Comparison

Tests whether Discrete Alignment Theory and H3-Hybrid principles
provide predictive advantage for atmospheric dynamics.
"""

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from dat_ml.core.device import DEVICE, HARDWARE
from dat_ml.core.config import ExperimentConfig, EXPERIMENTS_DIR
from dat_ml.data.netcdf_loader import ERA5RegionalDataset, ERA5PatchDataset
from dat_ml.data.loader import create_dataloaders
from dat_ml.models.h3_network import DAT_H3_Predictor
from dat_ml.models.baseline import StandardMLP, StandardTransformer, create_matched_baseline
from dat_ml.losses.topological import CombinedTopologicalLoss
from dat_ml.compare import ABComparison


def run_regional_comparison():
    """
    Compare models on regional Z500 prediction task.

    Task: Predict regional mean Z500 7 days ahead from 30 days of history.
    """
    print("=" * 70)
    print("ERA5 Z500 REGIONAL PREDICTION: DAT-H3 vs BASELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Hardware: {HARDWARE.total_cores} cores, {HARDWARE.memory_gb}GB RAM")
    print()

    # Load data
    nc_path = Path('/Users/bryan/Chaos_analogies/quasicrystal-chaos-analogies/phason_brain/era5_z500_2023_2025.nc')

    print("Loading ERA5 data...")
    dataset = ERA5RegionalDataset(
        nc_path,
        variable='z',
        sequence_length=30,  # 30 days history
        horizon=7,           # Predict 7 days ahead
        normalize=True
    )

    print(f"Regions: {dataset.get_region_names()}")
    print(f"Total samples: {len(dataset)}")
    print(f"Input: {dataset.sequence_length} days × {dataset.get_input_dim()} regions")
    print(f"Output: {dataset.get_input_dim()} regional values")
    print()

    # Create dataloaders
    config = ExperimentConfig()
    config.training.batch_size = 64
    config.training.epochs = 100
    config.training.learning_rate = 1e-3

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, config.data, config.training
    )

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test (BLIND): {len(test_loader.dataset)}")

    # Input dimension: sequence_length * num_regions (flattened)
    sample_x, sample_y = dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()

    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print()

    # Create models
    print("Creating models...")

    # DAT-H3 Model - smaller for fair comparison
    dat_h3 = DAT_H3_Predictor(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=output_dim,
        num_layers=2,
        use_attention=False,
        use_h3_hybrid=True
    ).to(DEVICE)

    dat_h3_params = sum(p.numel() for p in dat_h3.parameters())
    print(f"DAT-H3 parameters: {dat_h3_params:,}")

    # Baseline: MLP sized to match DAT-H3 params
    # DAT-H3 has ~787K params. With input=180, output=6:
    # hidden=256, layers=4: ~(180*256) + 3*(256*1024+256*256) + 256*6 ≈ 800K
    baseline = StandardMLP(
        input_dim=input_dim,
        hidden_dim=280,
        output_dim=output_dim,
        num_layers=4,
        dropout=0.1
    ).to(DEVICE)

    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Parameter ratio: {dat_h3_params/baseline_params:.2f}x")
    print()

    # Run comparison
    experiment_name = f"era5_regional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    comparison = ABComparison(experiment_name, config)

    # We'll manually run to have more control
    print("-" * 70)
    print("TRAINING DAT-H3 MODEL")
    print("-" * 70)

    dat_h3_losses = train_model(
        dat_h3, train_loader, val_loader, config,
        use_topological_loss=True, name="DAT-H3"
    )

    print()
    print("-" * 70)
    print("TRAINING BASELINE MODEL")
    print("-" * 70)

    baseline_losses = train_model(
        baseline, train_loader, val_loader, config,
        use_topological_loss=False, name="Baseline"
    )

    # BLIND TEST
    print()
    print("=" * 70)
    print("BLIND TEST EVALUATION")
    print("=" * 70)

    dat_h3_results = evaluate_model(dat_h3, test_loader, "DAT-H3")
    baseline_results = evaluate_model(baseline, test_loader, "Baseline")

    # Compare
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    compare_results(dat_h3_results, baseline_results)

    # Cleanup
    dataset.close()


def train_model(model, train_loader, val_loader, config, use_topological_loss, name):
    """Train a single model."""
    if use_topological_loss:
        criterion = CombinedTopologicalLoss(
            prediction_weight=1.0,
            localization_weight=0.05,
            energy_weight=0.05,
            coordination_weight=0.05,
            manifold_weight=0.02,
            symmetry_weight=0.02
        )
    else:
        criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.training.epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(config.training.epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Flatten sequence: (batch, seq, features) -> (batch, seq*features)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            output = model(x)
            pred = output['prediction']

            if use_topological_loss:
                losses = criterion(pred, y, output)
                loss = losses['total']
            else:
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

                output = model(x)
                pred = output['prediction']

                if use_topological_loss:
                    losses = criterion(pred, y, output)
                    loss = losses['total']
                else:
                    loss = criterion(pred.squeeze(), y)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        # Progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return {'best_val_loss': best_val_loss}


def evaluate_model(model, test_loader, name):
    """Evaluate on blind test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            output = model(x)
            pred = output['prediction']

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Metrics
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Correlation (average over regions)
    corrs = []
    for i in range(preds.shape[1]):
        c = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        corrs.append(c)
    mean_corr = np.mean(corrs)

    print(f"\n{name} Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  Corr: {mean_corr:.6f}")

    return {
        'name': name,
        'preds': preds,
        'targets': targets,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'corr': mean_corr
    }


def compare_results(dat_h3, baseline):
    """Statistical comparison."""
    import scipy.stats as stats

    print(f"\n{'Metric':<10} {'DAT-H3':>12} {'Baseline':>12} {'Diff':>12} {'Better':>10}")
    print("-" * 60)

    metrics = ['rmse', 'mae', 'r2', 'corr']
    for m in metrics:
        d = dat_h3[m]
        b = baseline[m]
        diff = d - b

        # For RMSE/MAE lower is better; for R²/corr higher is better
        if m in ['rmse', 'mae']:
            better = "DAT-H3" if d < b else "Baseline" if b < d else "Tie"
            diff = -diff  # Make positive = DAT-H3 better
        else:
            better = "DAT-H3" if d > b else "Baseline" if b > d else "Tie"

        print(f"{m:<10} {d:>12.6f} {b:>12.6f} {diff:>+12.6f} {better:>10}")

    # Statistical test on errors
    dat_errors = np.abs(dat_h3['preds'] - dat_h3['targets']).flatten()
    base_errors = np.abs(baseline['preds'] - baseline['targets']).flatten()

    t_stat, t_pval = stats.ttest_rel(base_errors, dat_errors)

    print()
    print(f"Paired t-test on absolute errors:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pval:.6f}")

    if t_pval < 0.05:
        if t_stat > 0:
            print(f"  → DAT-H3 significantly better (p < 0.05)")
        else:
            print(f"  → Baseline significantly better (p < 0.05)")
    else:
        print(f"  → No significant difference (p >= 0.05)")

    # Verdict
    print()
    print("=" * 60)
    wins = sum([
        dat_h3['rmse'] < baseline['rmse'],
        dat_h3['mae'] < baseline['mae'],
        dat_h3['r2'] > baseline['r2'],
        dat_h3['corr'] > baseline['corr']
    ])

    if wins >= 3 and t_pval < 0.05 and t_stat > 0:
        print("VERDICT: DAT-H3 provides significant predictive advantage")
    elif wins >= 3:
        print("VERDICT: DAT-H3 shows improvement but not statistically significant")
    elif wins <= 1:
        print("VERDICT: DAT-H3 does NOT improve over baseline")
    else:
        print("VERDICT: Mixed results - no clear winner")
    print("=" * 60)


if __name__ == '__main__':
    run_regional_comparison()
