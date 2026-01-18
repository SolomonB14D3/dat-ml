"""
A/B Comparison Framework.

Trains DAT-H3 model and baseline in parallel on identical data splits,
then performs rigorous statistical comparison to determine if the
theoretical framework provides predictive advantage.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import scipy.stats as stats

from .core.device import DEVICE, HARDWARE
from .core.config import ExperimentConfig, DATA_RAW, EXPERIMENTS_DIR
from .data.loader import auto_detect_dataset, create_dataloaders
from .models.h3_network import DAT_H3_Predictor
from .models.baseline import StandardMLP, StandardTransformer, MatchedBaseline, create_matched_baseline
from .losses.topological import CombinedTopologicalLoss
from .evaluation.blind_eval import BlindEvaluator, MetricCalculator


@dataclass
class ComparisonResult:
    """Results from A/B comparison."""
    dataset_name: str
    dat_h3_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    improvement: Dict[str, float]  # Positive = DAT-H3 is better
    p_values: Dict[str, float]
    is_significant: Dict[str, bool]
    dat_h3_params: int
    baseline_params: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"COMPARISON RESULTS: {self.dataset_name}",
            f"{'='*60}",
            f"Timestamp: {self.timestamp}",
            f"\nModel Parameters:",
            f"  DAT-H3:   {self.dat_h3_params:,}",
            f"  Baseline: {self.baseline_params:,}",
            f"\nBlind Test Metrics:",
            f"{'Metric':<15} {'DAT-H3':>12} {'Baseline':>12} {'Δ':>10} {'Sig?':>6}"
        ]

        for metric in self.dat_h3_metrics:
            dat_val = self.dat_h3_metrics[metric]
            base_val = self.baseline_metrics[metric]
            imp = self.improvement.get(metric, 0)
            sig = "YES" if self.is_significant.get(metric, False) else "no"

            # Format improvement with direction
            if metric in ['r2', 'correlation']:
                imp_str = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
            else:
                imp_str = f"-{abs(imp):.4f}" if imp > 0 else f"+{abs(imp):.4f}"

            lines.append(f"{metric:<15} {dat_val:>12.6f} {base_val:>12.6f} {imp_str:>10} {sig:>6}")

        # Overall verdict
        sig_improvements = sum(1 for v in self.is_significant.values() if v)
        total_metrics = len(self.is_significant)

        lines.extend([
            f"\n{'='*60}",
            f"VERDICT: {sig_improvements}/{total_metrics} metrics show significant improvement",
        ])

        if sig_improvements > total_metrics / 2:
            lines.append("DAT-H3 framework DOES provide predictive advantage")
        elif sig_improvements > 0:
            lines.append("DAT-H3 framework shows PARTIAL advantage")
        else:
            lines.append("DAT-H3 framework does NOT show clear advantage on this data")

        lines.append(f"{'='*60}\n")

        return "\n".join(lines)


class ABComparison:
    """
    Runs controlled A/B comparison between DAT-H3 and baseline models.
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[ExperimentConfig] = None
    ):
        self.experiment_name = experiment_name
        self.config = config or ExperimentConfig()

        self.experiment_dir = EXPERIMENTS_DIR / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ComparisonResult] = []

    def run(
        self,
        data_path: Path,
        baseline_type: str = 'matched'  # 'mlp', 'transformer', 'matched'
    ) -> ComparisonResult:
        """
        Run full A/B comparison on dataset.

        Args:
            data_path: Path to data file
            baseline_type: Type of baseline model

        Returns:
            ComparisonResult with statistical analysis
        """
        print(f"\n{'='*60}")
        print(f"A/B COMPARISON: DAT-H3 vs Baseline ({baseline_type})")
        print(f"Data: {data_path}")
        print(f"{'='*60}\n")

        # Load data
        print("Loading dataset...")
        dataset = auto_detect_dataset(data_path)
        print(f"Dataset size: {len(dataset)}")

        # Create dataloaders - SAME splits for both models
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            self.config.data,
            self.config.training
        )
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
              f"Test (BLIND): {len(test_loader.dataset)}")

        # Get input dimension
        sample_x, _ = dataset[0]
        input_dim = sample_x.numel()
        print(f"Input dimension: {input_dim}")

        # Create DAT-H3 model
        print("\nCreating DAT-H3 model...")
        dat_h3_model = DAT_H3_Predictor(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=1,
            num_layers=4,
            use_attention=True,
            use_h3_hybrid=True
        )
        dat_h3_params = sum(p.numel() for p in dat_h3_model.parameters())
        print(f"DAT-H3 parameters: {dat_h3_params:,}")

        # Create baseline model
        print("\nCreating baseline model...")
        if baseline_type == 'matched':
            baseline_model = create_matched_baseline(dat_h3_model, input_dim)
        elif baseline_type == 'transformer':
            baseline_model = StandardTransformer(input_dim, hidden_dim=128, num_layers=4)
        else:
            baseline_model = StandardMLP(input_dim, hidden_dim=128, num_layers=4)

        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"Baseline parameters: {baseline_params:,}")
        print(f"Parameter ratio: {dat_h3_params/baseline_params:.2f}x")

        # Train both models
        print("\n" + "-"*40)
        print("TRAINING DAT-H3 MODEL")
        print("-"*40)
        dat_h3_history = self._train_model(
            dat_h3_model, train_loader, val_loader, "dat_h3",
            use_topological_loss=True
        )

        print("\n" + "-"*40)
        print("TRAINING BASELINE MODEL")
        print("-"*40)
        baseline_history = self._train_model(
            baseline_model, train_loader, val_loader, "baseline",
            use_topological_loss=False  # No topological constraints
        )

        # BLIND EVALUATION
        print("\n" + "="*40)
        print("BLIND TEST EVALUATION")
        print("="*40)

        # Load best models
        dat_h3_model = self._load_best_model(dat_h3_model, "dat_h3")
        baseline_model = self._load_best_model(baseline_model, "baseline")

        # Evaluate on blind test set
        dat_h3_preds, dat_h3_targets = self._evaluate(dat_h3_model, test_loader)
        baseline_preds, baseline_targets = self._evaluate(baseline_model, test_loader)

        # Compute metrics
        dat_h3_metrics = MetricCalculator.compute_all(dat_h3_preds, dat_h3_targets)
        baseline_metrics = MetricCalculator.compute_all(baseline_preds, baseline_targets)

        # Statistical comparison
        comparison = self._statistical_comparison(
            dat_h3_preds, baseline_preds, dat_h3_targets,
            dat_h3_metrics, baseline_metrics
        )

        result = ComparisonResult(
            dataset_name=data_path.stem,
            dat_h3_metrics=dat_h3_metrics,
            baseline_metrics=baseline_metrics,
            improvement=comparison['improvement'],
            p_values=comparison['p_values'],
            is_significant=comparison['is_significant'],
            dat_h3_params=dat_h3_params,
            baseline_params=baseline_params
        )

        # Save results
        self._save_result(result)
        self.results.append(result)

        # Print summary
        print(result.summary())

        return result

    def _train_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        name: str,
        use_topological_loss: bool
    ) -> Dict:
        """Train a single model."""
        model = model.to(DEVICE)

        if use_topological_loss:
            criterion = CombinedTopologicalLoss()
        else:
            # Simple MSE loss for baseline
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.training.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': []}

        for epoch in range(self.config.training.epochs):
            # Train
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                output = model(x)
                pred = output['prediction']

                if use_topological_loss:
                    losses = criterion(pred, y, output)
                    loss = losses['total']
                else:
                    loss = criterion(pred.squeeze(), y.squeeze())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train'].append(train_loss)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = model(x)
                    pred = output['prediction']

                    if use_topological_loss:
                        losses = criterion(pred, y, output)
                        loss = losses['total']
                    else:
                        loss = criterion(pred.squeeze(), y.squeeze())

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history['val'].append(val_loss)

            scheduler.step()

            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model, name)
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        return history

    def _evaluate(
        self,
        model: nn.Module,
        loader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model on data loader."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                pred = output['prediction']

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)

    def _statistical_comparison(
        self,
        dat_h3_preds: np.ndarray,
        baseline_preds: np.ndarray,
        targets: np.ndarray,
        dat_h3_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        alpha: float = 0.05
    ) -> Dict:
        """Perform statistical comparison of predictions."""
        dat_h3_errors = np.abs(dat_h3_preds.flatten() - targets.flatten())
        baseline_errors = np.abs(baseline_preds.flatten() - targets.flatten())

        improvement = {}
        p_values = {}
        is_significant = {}

        # Compare each metric
        for metric in dat_h3_metrics:
            dat_val = dat_h3_metrics[metric]
            base_val = baseline_metrics[metric]

            # For r2/correlation, higher is better; for errors, lower is better
            if metric in ['r2', 'correlation']:
                improvement[metric] = dat_val - base_val
            else:
                improvement[metric] = base_val - dat_val  # Positive = DAT-H3 better

        # Paired t-test on errors
        t_stat, t_pval = stats.ttest_rel(baseline_errors, dat_h3_errors)
        p_values['error_ttest'] = float(t_pval)
        # One-sided: is DAT-H3 significantly better?
        is_significant['error'] = t_pval < alpha and t_stat > 0

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(
                baseline_errors - dat_h3_errors,
                alternative='greater'  # Baseline errors > DAT-H3 errors
            )
            p_values['error_wilcoxon'] = float(w_pval)
            is_significant['error_wilcoxon'] = w_pval < alpha
        except ValueError:
            p_values['error_wilcoxon'] = 1.0
            is_significant['error_wilcoxon'] = False

        # Bootstrap confidence interval for improvement
        n_bootstrap = 1000
        improvements = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(targets), len(targets), replace=True)
            dat_err = np.abs(dat_h3_preds.flatten()[idx] - targets.flatten()[idx]).mean()
            base_err = np.abs(baseline_preds.flatten()[idx] - targets.flatten()[idx]).mean()
            improvements.append(base_err - dat_err)

        ci_lower = np.percentile(improvements, 2.5)
        ci_upper = np.percentile(improvements, 97.5)
        p_values['bootstrap_ci'] = (ci_lower, ci_upper)
        is_significant['bootstrap'] = ci_lower > 0  # CI doesn't include 0

        # Per-metric significance (using bootstrap)
        for metric in ['rmse', 'mae', 'r2']:
            is_significant[metric] = is_significant.get('bootstrap', False)
            p_values[metric] = p_values.get('error_ttest', 1.0)

        return {
            'improvement': improvement,
            'p_values': p_values,
            'is_significant': is_significant
        }

    def _save_checkpoint(self, model: nn.Module, name: str):
        """Save model checkpoint."""
        torch.save(
            model.state_dict(),
            self.experiment_dir / f"{name}_best.pt"
        )

    def _load_best_model(self, model: nn.Module, name: str) -> nn.Module:
        """Load best model checkpoint."""
        model.load_state_dict(
            torch.load(self.experiment_dir / f"{name}_best.pt", map_location=DEVICE)
        )
        return model.to(DEVICE)

    def _save_result(self, result: ComparisonResult):
        """Save comparison result to disk."""
        result_dict = {
            'dataset_name': result.dataset_name,
            'dat_h3_metrics': result.dat_h3_metrics,
            'baseline_metrics': result.baseline_metrics,
            'improvement': result.improvement,
            'p_values': {k: str(v) for k, v in result.p_values.items()},
            'is_significant': result.is_significant,
            'dat_h3_params': result.dat_h3_params,
            'baseline_params': result.baseline_params,
            'timestamp': result.timestamp
        }

        with open(self.experiment_dir / 'comparison_result.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        # Also save human-readable summary
        with open(self.experiment_dir / 'comparison_summary.txt', 'w') as f:
            f.write(result.summary())


def run_comparison(
    data_path: Path,
    experiment_name: Optional[str] = None,
    baseline_type: str = 'matched'
) -> ComparisonResult:
    """
    Convenience function to run A/B comparison.

    Args:
        data_path: Path to dataset
        experiment_name: Name for experiment (auto-generated if None)
        baseline_type: 'matched', 'mlp', or 'transformer'

    Returns:
        ComparisonResult with full statistical analysis
    """
    if experiment_name is None:
        experiment_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    comparison = ABComparison(experiment_name)
    return comparison.run(Path(data_path), baseline_type)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='DAT-H3 vs Baseline Comparison')
    parser.add_argument('data_path', type=Path, help='Path to data file')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument(
        '--baseline', type=str, default='matched',
        choices=['matched', 'mlp', 'transformer'],
        help='Baseline model type'
    )

    args = parser.parse_args()
    run_comparison(args.data_path, args.name, args.baseline)


if __name__ == '__main__':
    main()
