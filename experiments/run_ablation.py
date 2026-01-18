#!/usr/bin/env python3
"""
Ablation Study: Why does DAT-H3 underperform?

Systematically disables components to identify what's hurting performance.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import scipy.stats as stats

from dat_ml.core.device import DEVICE, HARDWARE
from dat_ml.core.config import ExperimentConfig
from dat_ml.data.netcdf_loader import ERA5RegionalDataset
from dat_ml.data.loader import create_dataloaders
from dat_ml.models.h3_network import DAT_H3_Predictor
from dat_ml.models.baseline import StandardMLP
from dat_ml.losses.topological import CombinedTopologicalLoss
from dat_ml.transforms.e6_projection import DATTransform


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    # Model config
    use_e6_embedding: bool = True
    use_h3_projection: bool = True
    use_icosahedral_blocks: bool = True
    use_h3_hybrid: bool = True
    # Loss config
    use_topological_loss: bool = True
    localization_weight: float = 0.02
    energy_weight: float = 0.02
    coordination_weight: float = 0.02
    manifold_weight: float = 0.01
    symmetry_weight: float = 0.01


# Define ablation experiments
ABLATIONS = [
    # Baseline
    AblationConfig(
        name="baseline_mlp",
        description="Standard MLP (no DAT components)",
        use_e6_embedding=False,
        use_h3_projection=False,
        use_icosahedral_blocks=False,
        use_h3_hybrid=False,
        use_topological_loss=False,
    ),

    # Full DAT-H3
    AblationConfig(
        name="full_dat_h3",
        description="Full DAT-H3 with all components",
    ),

    # Remove E6 embedding
    AblationConfig(
        name="no_e6_embedding",
        description="DAT-H3 without E6 embedding (direct input)",
        use_e6_embedding=False,
    ),

    # Remove H3 Hybrid compression
    AblationConfig(
        name="no_h3_hybrid",
        description="DAT-H3 without H3 Hybrid compression",
        use_h3_hybrid=False,
    ),

    # Remove topological losses entirely
    AblationConfig(
        name="no_topo_loss",
        description="DAT-H3 architecture with standard MSE loss",
        use_topological_loss=False,
    ),

    # Remove localization loss only
    AblationConfig(
        name="no_localization",
        description="No phason localization loss",
        localization_weight=0.0,
    ),

    # Remove energy loss only
    AblationConfig(
        name="no_energy",
        description="No energy depth loss",
        energy_weight=0.0,
    ),

    # Remove coordination loss only
    AblationConfig(
        name="no_coordination",
        description="No coordination loss",
        coordination_weight=0.0,
    ),

    # Remove manifold/symmetry losses
    AblationConfig(
        name="no_manifold_symmetry",
        description="No manifold or symmetry losses",
        manifold_weight=0.0,
        symmetry_weight=0.0,
    ),

    # Only prediction loss (but keep DAT architecture)
    AblationConfig(
        name="dat_arch_mse_only",
        description="DAT architecture, MSE loss only",
        localization_weight=0.0,
        energy_weight=0.0,
        coordination_weight=0.0,
        manifold_weight=0.0,
        symmetry_weight=0.0,
    ),

    # Topological losses on standard MLP
    AblationConfig(
        name="mlp_with_topo_loss",
        description="Standard MLP with topological losses",
        use_e6_embedding=False,
        use_h3_projection=False,
        use_icosahedral_blocks=False,
        use_h3_hybrid=False,
        use_topological_loss=True,
    ),
]


class AblationModel(nn.Module):
    """
    Configurable model for ablation studies.
    Can enable/disable individual DAT-H3 components.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        config: AblationConfig
    ):
        super().__init__()
        self.config = config

        # E6 embedding (optional)
        if config.use_e6_embedding:
            from dat_ml.transforms.e6_projection import E6Embedding, H3Projection
            self.e6_embed = E6Embedding(input_dim, learnable=True)
            if config.use_h3_projection:
                self.h3_proj = H3Projection(phason_steering=True)
                embed_out_dim = 6  # parallel(3) + perp(3)
            else:
                self.h3_proj = None
                embed_out_dim = 6  # E6 dimension
        else:
            self.e6_embed = None
            self.h3_proj = None
            embed_out_dim = input_dim

        # Main backbone
        if config.use_icosahedral_blocks:
            from dat_ml.models.h3_network import IcosahedralBlock
            self.input_proj = nn.Linear(embed_out_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                IcosahedralBlock(hidden_dim) for _ in range(3)
            ])
        else:
            # Standard MLP blocks
            self.input_proj = nn.Linear(embed_out_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for _ in range(3)
            ])

        # H3 Hybrid compression (optional)
        if config.use_h3_hybrid:
            from dat_ml.transforms.h3_hybrid import H3HybridTransform
            self.h3_hybrid = H3HybridTransform(
                hidden_dim,
                use_energy_landscape=True,
                use_coordination=True
            )
            final_dim = self.h3_hybrid.output_dim
        else:
            self.h3_hybrid = None
            final_dim = hidden_dim

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Linear(final_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Flatten if needed
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.view(B, -1)

        # E6 embedding
        parallel, perp = None, None
        if self.e6_embed is not None:
            e6 = self.e6_embed(x)
            if self.h3_proj is not None:
                parallel, perp = self.h3_proj(e6)
                x = torch.cat([parallel, perp], dim=-1)
            else:
                x = e6

        # Project to hidden dim
        x = self.input_proj(x)

        # Backbone
        for block in self.blocks:
            if hasattr(block, 'forward'):
                x = block(x)
            else:
                x = x + block(x)

        # H3 Hybrid
        energy, basin_probs, shell_occupancy = None, None, None
        features = x
        if self.h3_hybrid is not None:
            h3_out = self.h3_hybrid(x)
            features = h3_out['features']
            energy = h3_out.get('energy')
            basin_probs = h3_out.get('basin_probs')
            shell_occupancy = h3_out.get('shell_occupancy')

        # Output
        pred = self.output_head(features)

        return {
            'prediction': pred,
            'features': features,
            'parallel': parallel,
            'perp': perp,
            'energy': energy,
            'basin_probs': basin_probs,
            'shell_occupancy': shell_occupancy
        }


def train_ablation(model, train_loader, val_loader, config: AblationConfig, epochs=100):
    """Train model with specified ablation config."""
    model = model.to(DEVICE)

    if config.use_topological_loss:
        criterion = CombinedTopologicalLoss(
            prediction_weight=1.0,
            localization_weight=config.localization_weight,
            energy_weight=config.energy_weight,
            coordination_weight=config.coordination_weight,
            manifold_weight=config.manifold_weight,
            symmetry_weight=config.symmetry_weight
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            out = model(x)
            pred = out['prediction']

            if config.use_topological_loss:
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

                if config.use_topological_loss:
                    losses = criterion(pred, y, out)
                    val_loss += losses['total'].item()
                else:
                    val_loss += criterion(pred.squeeze(), y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model, best_val_loss


def evaluate_model(model, test_loader):
    """Evaluate on blind test set."""
    model.eval()
    all_preds, all_targets = [], []

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

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': preds, 'targets': targets}


def main():
    print("=" * 70)
    print("ABLATION STUDY: Why does DAT-H3 underperform?")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load ERA5 data
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
    hidden_dim = 64

    print(f"Data: {len(dataset)} samples, input={input_dim}, output={output_dim}")

    # Run ablations
    results = []

    for ablation in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"ABLATION: {ablation.name}")
        print(f"  {ablation.description}")
        print("=" * 60)

        # Create model
        model = AblationModel(input_dim, hidden_dim, output_dim, ablation)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Train
        model, val_loss = train_ablation(
            model, train_loader, val_loader, ablation, epochs=100
        )
        print(f"  Best val loss: {val_loss:.6f}")

        # Evaluate
        metrics = evaluate_model(model, test_loader)
        print(f"  Test RMSE: {metrics['rmse']:.6f}")
        print(f"  Test R²:   {metrics['r2']:.6f}")

        results.append({
            'name': ablation.name,
            'description': ablation.description,
            'params': n_params,
            'val_loss': val_loss,
            **{k: v for k, v in metrics.items() if k not in ['preds', 'targets']}
        })

    # Summary table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Ablation':<25} {'Params':>10} {'RMSE':>10} {'R²':>10} {'Δ from baseline':>15}")
    print("-" * 80)

    baseline_r2 = results[0]['r2']  # First result is baseline MLP

    for r in sorted(results, key=lambda x: -x['r2']):
        delta = r['r2'] - baseline_r2
        delta_str = f"{delta:+.4f}" if delta != 0 else "---"
        print(f"{r['name']:<25} {r['params']:>10,} {r['rmse']:>10.4f} {r['r2']:>10.4f} {delta_str:>15}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find best and worst
    best = max(results, key=lambda x: x['r2'])
    worst = min(results, key=lambda x: x['r2'])

    print(f"\nBest performer:  {best['name']} (R²={best['r2']:.4f})")
    print(f"Worst performer: {worst['name']} (R²={worst['r2']:.4f})")

    # Compare specific ablations
    full_dat = next(r for r in results if r['name'] == 'full_dat_h3')
    no_topo = next(r for r in results if r['name'] == 'no_topo_loss')
    dat_mse = next(r for r in results if r['name'] == 'dat_arch_mse_only')

    print(f"\nEffect of topological losses:")
    print(f"  Full DAT-H3:     R²={full_dat['r2']:.4f}")
    print(f"  DAT + MSE only:  R²={dat_mse['r2']:.4f}")
    print(f"  → Topo losses {'help' if full_dat['r2'] > dat_mse['r2'] else 'HURT'} by {abs(full_dat['r2'] - dat_mse['r2']):.4f}")

    baseline = results[0]
    print(f"\nEffect of DAT architecture (vs baseline MLP):")
    print(f"  Baseline MLP:    R²={baseline['r2']:.4f}")
    print(f"  DAT + MSE only:  R²={dat_mse['r2']:.4f}")
    print(f"  → DAT architecture {'helps' if dat_mse['r2'] > baseline['r2'] else 'HURTS'} by {abs(dat_mse['r2'] - baseline['r2']):.4f}")

    # Cleanup
    dataset.close()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
