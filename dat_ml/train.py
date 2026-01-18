"""
Main training script for DAT-H3 prediction models.
Optimized for Apple M3 Ultra (MPS backend, 96GB unified memory).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from datetime import datetime
import argparse

from .core.device import DEVICE, HARDWARE, get_optimal_workers
from .core.config import (
    ExperimentConfig, DataConfig, TrainingConfig, DATConfig,
    DATA_RAW, EXPERIMENTS_DIR
)
from .data.loader import (
    auto_detect_dataset, create_dataloaders,
    TimeSeriesDataset, PhysicalSystemDataset, TabularDataset
)
from .models.h3_network import DAT_H3_Predictor, EnsemblePredictor
from .losses.topological import CombinedTopologicalLoss
from .evaluation.blind_eval import BlindEvaluator, LinearBenchmark, PersistenceBenchmark


class Trainer:
    """
    Main training loop with DAT-H3 specific optimizations.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        experiment_name: Optional[str] = None
    ):
        self.model = model.to(DEVICE)
        self.config = config

        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = EXPERIMENTS_DIR / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(self.experiment_dir / 'config.json')

        # Loss function
        self.criterion = CombinedTopologicalLoss()

        # Optimizer with M3 Ultra optimizations
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=1e-5,
            fused=False  # Fused not yet supported on MPS
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0, 'prediction': 0.0,
            'localization': 0.0, 'energy_depth': 0.0
        }
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Forward pass
            output = self.model(x)
            pred = output['prediction']

            # Compute loss
            losses = self.criterion(pred, y, output)

            # Backward pass with gradient accumulation
            loss = losses['total'] / self.config.training.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Run validation."""
        self.model.eval()
        epoch_losses = {'total': 0.0, 'prediction': 0.0}
        all_preds = []
        all_targets = []
        num_batches = 0

        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            output = self.model(x)
            pred = output['prediction']

            losses = self.criterion(pred, y, output)

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        return epoch_losses, preds, targets

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict:
        """
        Full training loop.

        Note: test_loader is BLIND - only used once at the very end.
        """
        print(f"Training on {DEVICE} with {HARDWARE.memory_gb}GB unified memory")
        print(f"Experiment: {self.experiment_name}")
        print("-" * 50)

        for epoch in range(self.config.training.epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_history.append(train_losses)

            # Validate
            val_losses, val_preds, val_targets = self.validate(val_loader)
            self.val_history.append(val_losses)

            # Learning rate step
            self.scheduler.step()

            # Print progress
            print(f"Epoch {epoch+1}/{self.config.training.epochs}")
            print(f"  Train Loss: {train_losses['total']:.6f}")
            print(f"  Val Loss:   {val_losses['total']:.6f}")

            # Early stopping check
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save final model
        self._save_checkpoint('final_model.pt')

        # Save training history
        self._save_history()

        # BLIND TEST EVALUATION
        print("\n" + "=" * 50)
        print("BLIND TEST EVALUATION")
        print("=" * 50)

        # Load best model for final evaluation
        self._load_checkpoint('best_model.pt')

        # Evaluate on blind test set
        evaluator = BlindEvaluator(self.experiment_dir)
        test_result = evaluator.evaluate_model(
            self.model,
            test_loader,
            split='test',
            model_name='DAT_H3_Predictor',
            dataset_name=self.experiment_name,
            device=DEVICE
        )

        print("\nBlind Test Results:")
        for metric, value in test_result.metrics.items():
            print(f"  {metric}: {value:.6f}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'test_result': test_result
        }

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, self.experiment_dir / filename)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.experiment_dir / filename, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def run_experiment(
    data_path: Path,
    experiment_name: Optional[str] = None,
    config: Optional[ExperimentConfig] = None
) -> Dict:
    """
    Run a complete experiment: load data, train model, blind evaluation.
    """
    config = config or ExperimentConfig()

    # Auto-detect and load dataset
    print(f"Loading data from {data_path}")
    dataset = auto_detect_dataset(data_path)
    print(f"Dataset type: {type(dataset).__name__}, size: {len(dataset)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        config.data,
        config.training
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test (BLIND): {len(test_loader.dataset)}")

    # Determine input dimension from data
    sample_x, _ = dataset[0]
    input_dim = sample_x.numel()
    print(f"Input dimension: {input_dim}")

    # Create model
    model = DAT_H3_Predictor(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=1,
        num_layers=4,
        use_attention=True,
        use_h3_hybrid=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    trainer = Trainer(model, config, experiment_name)
    results = trainer.train(train_loader, val_loader, test_loader)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='DAT-H3 ML Training')
    parser.add_argument('data_path', type=Path, help='Path to data file or directory')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    config = ExperimentConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    run_experiment(args.data_path, args.name, config)


if __name__ == '__main__':
    main()
