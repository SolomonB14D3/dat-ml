#!/usr/bin/env python3
"""
Pre-train on DAT simulation data, then transfer to weather prediction.

Tests whether learning explicit DAT patterns helps generalize to other domains.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Before importing torch

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from dat_ml.core.device import DEVICE, HARDWARE
from dat_ml.core.config import ExperimentConfig
from dat_ml.data.dat_pretrain_loader import DATSimulationDataset, load_dat_pretrain_data
from dat_ml.data.netcdf_loader import ERA5RegionalDataset
from dat_ml.data.loader import create_dataloaders
from dat_ml.models.h3_network import DAT_H3_Predictor
from dat_ml.models.baseline import StandardMLP
from dat_ml.losses.topological import CombinedTopologicalLoss


def pretrain_on_dat(model, epochs=50, lr=1e-3):
    """
    Pre-train model on DAT simulation data.

    Teaches the model what E6/H3 patterns look like.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: PRE-TRAINING ON DAT SIMULATION DATA")
    print("=" * 60)

    # Load DAT data - dynamics prediction task
    dataset = load_dat_pretrain_data(
        task='predict_dynamics',
        sequence_length=16
    )

    print(f"DAT data: {len(dataset)} samples")
    print(f"Input dim: {dataset.get_input_dim()} (6D offset + 3D pos + 1 vorticity)")
    print(f"Output dim: {dataset.get_output_dim()} (next state)")

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    # We need to adapt the model input for pre-training
    # DAT data has different dims than ERA5, so we'll use a projection layer
    pretrain_input_dim = dataset.sequence_length * dataset.get_input_dim()
    pretrain_output_dim = dataset.get_output_dim()

    # Create adapter layers
    input_adapter = nn.Linear(pretrain_input_dim, 180).to(DEVICE)  # Match ERA5 input
    output_adapter = nn.Linear(6, pretrain_output_dim).to(DEVICE)  # Match DAT output

    # Combined forward
    def pretrain_forward(x):
        x_adapted = input_adapter(x)
        out = model(x_adapted)
        pred = out['prediction']
        return output_adapter(pred)

    # Training with full topological losses (now MPS-compatible)
    criterion = CombinedTopologicalLoss(
        prediction_weight=1.0,
        localization_weight=0.1,  # Higher weight during pre-training
        energy_weight=0.1,
        coordination_weight=0.1,
        manifold_weight=0.05,
        symmetry_weight=0.05
    )
    mse_loss = nn.MSELoss()

    params = list(model.parameters()) + list(input_adapter.parameters()) + list(output_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        input_adapter.train()
        output_adapter.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)  # Flatten sequence

            optimizer.zero_grad()

            # Forward through adapted model
            x_adapted = input_adapter(x)
            out = model(x_adapted)
            pred_intermediate = out['prediction']
            pred = output_adapter(pred_intermediate)

            # Loss: prediction + topological (on intermediate)
            pred_loss = mse_loss(pred, y)

            # Topological loss on model's internal representation
            topo_losses = criterion(pred_intermediate, y[:, :6], out)  # Use offset_6d as pseudo-target
            loss = pred_loss + 0.1 * (topo_losses['total'] - topo_losses['prediction'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        input_adapter.eval()
        output_adapter.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)

                x_adapted = input_adapter(x)
                out = model(x_adapted)
                pred = output_adapter(out['prediction'])

                val_loss += mse_loss(pred, y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.6f}")
    return model


def finetune_on_era5(model, train_loader, val_loader, epochs=100, lr=1e-4, use_topological_loss=True, name="Model"):
    """Fine-tune on ERA5 data."""
    print(f"\n--- Fine-tuning {name} ---")

    if use_topological_loss:
        criterion = CombinedTopologicalLoss(
            prediction_weight=1.0,
            localization_weight=0.02,
            energy_weight=0.02,
            coordination_weight=0.02,
            manifold_weight=0.01,
            symmetry_weight=0.01
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            out = model(x)
            pred = out['prediction']

            if use_topological_loss:
                losses = criterion(pred, y, out)
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

                out = model(x)
                pred = out['prediction']

                if use_topological_loss:
                    losses = criterion(pred, y, out)
                    loss = losses['total']
                else:
                    loss = criterion(pred.squeeze(), y)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model, best_val_loss


def evaluate_model(model, test_loader, name):
    """Evaluate on blind test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            out = model(x)
            pred = out['prediction']

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    corrs = []
    for i in range(preds.shape[1]):
        c = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        corrs.append(c)
    mean_corr = np.mean(corrs)

    return {
        'name': name,
        'preds': preds,
        'targets': targets,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'corr': mean_corr
    }


def main():
    print("=" * 70)
    print("PRE-TRAINING TRANSFER EXPERIMENT")
    print("DAT Simulation → ERA5 Weather Prediction")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Hardware: {HARDWARE.total_cores} cores, {HARDWARE.memory_gb}GB RAM")

    # Load ERA5 data
    print("\nLoading ERA5 data...")
    nc_path = Path('/Users/bryan/Chaos_analogies/quasicrystal-chaos-analogies/phason_brain/era5_z500_2023_2025.nc')

    era5_dataset = ERA5RegionalDataset(
        nc_path,
        variable='z',
        sequence_length=30,
        horizon=7,
        normalize=True,
        preload_device='mps'  # Pre-load to GPU memory
    )

    config = ExperimentConfig()
    config.training.batch_size = 64
    config.training.epochs = 100

    train_loader, val_loader, test_loader = create_dataloaders(
        era5_dataset, config.data, config.training
    )

    sample_x, sample_y = era5_dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()

    print(f"ERA5: {len(era5_dataset)} samples, input={input_dim}, output={output_dim}")

    # ============================================
    # MODEL 1: DAT-H3 with pre-training
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL 1: DAT-H3 WITH PRE-TRAINING")
    print("=" * 60)

    dat_h3_pretrained = DAT_H3_Predictor(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=output_dim,
        num_layers=2,
        use_attention=False,
        use_h3_hybrid=True  # Now MPS-compatible
    ).to(DEVICE)

    # Pre-train on DAT data
    dat_h3_pretrained = pretrain_on_dat(dat_h3_pretrained, epochs=50, lr=1e-3)

    # Fine-tune on ERA5
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING ON ERA5")
    print("=" * 60)

    dat_h3_pretrained, _ = finetune_on_era5(
        dat_h3_pretrained, train_loader, val_loader,
        epochs=100, lr=1e-4, use_topological_loss=True, name="DAT-H3 (pretrained)"
    )

    # ============================================
    # MODEL 2: DAT-H3 without pre-training
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL 2: DAT-H3 WITHOUT PRE-TRAINING")
    print("=" * 60)

    dat_h3_scratch = DAT_H3_Predictor(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=output_dim,
        num_layers=2,
        use_attention=False,
        use_h3_hybrid=True  # Now MPS-compatible
    ).to(DEVICE)

    dat_h3_scratch, _ = finetune_on_era5(
        dat_h3_scratch, train_loader, val_loader,
        epochs=100, lr=1e-4, use_topological_loss=True, name="DAT-H3 (scratch)"
    )

    # ============================================
    # MODEL 3: Baseline MLP (no DAT, no pretrain)
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL 3: BASELINE MLP")
    print("=" * 60)

    baseline = StandardMLP(
        input_dim=input_dim,
        hidden_dim=280,
        output_dim=output_dim,
        num_layers=4,
        dropout=0.1
    ).to(DEVICE)

    baseline, _ = finetune_on_era5(
        baseline, train_loader, val_loader,
        epochs=100, lr=1e-3, use_topological_loss=False, name="Baseline MLP"
    )

    # ============================================
    # BLIND EVALUATION
    # ============================================
    print("\n" + "=" * 70)
    print("BLIND TEST EVALUATION")
    print("=" * 70)

    results_pretrained = evaluate_model(dat_h3_pretrained, test_loader, "DAT-H3 (pretrained)")
    results_scratch = evaluate_model(dat_h3_scratch, test_loader, "DAT-H3 (scratch)")
    results_baseline = evaluate_model(baseline, test_loader, "Baseline MLP")

    # Print results
    print(f"\n{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Corr':>10}")
    print("-" * 70)
    for r in [results_pretrained, results_scratch, results_baseline]:
        print(f"{r['name']:<25} {r['rmse']:>10.6f} {r['mae']:>10.6f} {r['r2']:>10.6f} {r['corr']:>10.6f}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    import scipy.stats as stats

    # Compare pretrained vs scratch
    err_pretrained = np.abs(results_pretrained['preds'] - results_pretrained['targets']).flatten()
    err_scratch = np.abs(results_scratch['preds'] - results_scratch['targets']).flatten()
    err_baseline = np.abs(results_baseline['preds'] - results_baseline['targets']).flatten()

    t1, p1 = stats.ttest_rel(err_scratch, err_pretrained)
    print(f"\nPretrained vs Scratch:")
    print(f"  t={t1:.4f}, p={p1:.6f}")
    if p1 < 0.05 and t1 > 0:
        print(f"  → Pre-training HELPS (p < 0.05)")
    elif p1 < 0.05 and t1 < 0:
        print(f"  → Pre-training HURTS (p < 0.05)")
    else:
        print(f"  → No significant difference")

    t2, p2 = stats.ttest_rel(err_baseline, err_pretrained)
    print(f"\nPretrained vs Baseline:")
    print(f"  t={t2:.4f}, p={p2:.6f}")
    if p2 < 0.05 and t2 > 0:
        print(f"  → Pretrained DAT-H3 BEATS baseline (p < 0.05)")
    elif p2 < 0.05 and t2 < 0:
        print(f"  → Baseline BEATS pretrained DAT-H3 (p < 0.05)")
    else:
        print(f"  → No significant difference")

    # Overall verdict
    print("\n" + "=" * 70)
    pretrain_helps = p1 < 0.05 and t1 > 0
    beats_baseline = p2 < 0.05 and t2 > 0

    if pretrain_helps and beats_baseline:
        print("VERDICT: Pre-training on DAT data enables transfer learning!")
        print("         DAT-H3 principles DO generalize when properly taught.")
    elif pretrain_helps:
        print("VERDICT: Pre-training helps DAT-H3, but still doesn't beat baseline.")
        print("         The inductive bias may need more data or different architecture.")
    elif beats_baseline:
        print("VERDICT: DAT-H3 beats baseline, but pre-training didn't help.")
        print("         The architecture itself provides the advantage.")
    else:
        print("VERDICT: Neither pre-training nor DAT-H3 architecture helps on this task.")
        print("         Weather prediction may need different inductive biases.")
    print("=" * 70)

    # Cleanup
    era5_dataset.close()


if __name__ == '__main__':
    main()
