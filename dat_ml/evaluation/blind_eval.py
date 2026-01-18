"""
Blind Evaluation Framework.

Ensures rigorous comparison between DAT-H3 models and analytical benchmarks:
- Strict train/val/test separation (test is BLIND)
- Statistical significance testing
- Multiple evaluation metrics
- Benchmark comparison tracking
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import scipy.stats as stats


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    dataset_name: str
    split: str  # 'train', 'val', or 'test' (blind)
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'split': self.split,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }


@dataclass
class BenchmarkComparison:
    """Comparison between model and analytical benchmark."""
    model_result: EvaluationResult
    benchmark_result: EvaluationResult
    improvement: Dict[str, float]  # Positive = model is better
    p_values: Dict[str, float]  # Statistical significance
    is_significant: Dict[str, bool]  # p < 0.05


class MetricCalculator:
    """Calculate various prediction metrics."""

    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((pred - target) ** 2))

    @staticmethod
    def rmse(pred: np.ndarray, target: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((pred - target) ** 2)))

    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(pred - target)))

    @staticmethod
    def mape(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        return float(np.mean(np.abs((target - pred) / (target + epsilon))) * 100)

    @staticmethod
    def r2(pred: np.ndarray, target: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-8))

    @staticmethod
    def correlation(pred: np.ndarray, target: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        return float(np.corrcoef(pred.flatten(), target.flatten())[0, 1])

    @staticmethod
    def max_error(pred: np.ndarray, target: np.ndarray) -> float:
        """Maximum absolute error."""
        return float(np.max(np.abs(pred - target)))

    @staticmethod
    def percentile_error(pred: np.ndarray, target: np.ndarray, p: int = 95) -> float:
        """Percentile of absolute errors."""
        return float(np.percentile(np.abs(pred - target), p))

    @classmethod
    def compute_all(cls, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'mse': cls.mse(pred, target),
            'rmse': cls.rmse(pred, target),
            'mae': cls.mae(pred, target),
            'mape': cls.mape(pred, target),
            'r2': cls.r2(pred, target),
            'correlation': cls.correlation(pred, target),
            'max_error': cls.max_error(pred, target),
            'p95_error': cls.percentile_error(pred, target, 95)
        }


class AnalyticalBenchmark:
    """
    Base class for analytical/equation-based benchmarks.
    Subclass this for specific domains.
    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions using analytical method."""
        raise NotImplementedError

    def evaluate(
        self,
        x: np.ndarray,
        target: np.ndarray,
        dataset_name: str
    ) -> EvaluationResult:
        """Evaluate benchmark on data."""
        pred = self.predict(x)
        metrics = MetricCalculator.compute_all(pred, target)
        return EvaluationResult(
            model_name=self.name,
            dataset_name=dataset_name,
            split='test',
            metrics=metrics,
            predictions=pred,
            targets=target
        )


class LinearBenchmark(AnalyticalBenchmark):
    """Simple linear regression benchmark."""

    def __init__(self):
        super().__init__('linear_regression')
        self.weights = None
        self.bias = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit linear model."""
        # Add bias term
        X = np.column_stack([x.reshape(len(x), -1), np.ones(len(x))])
        # Least squares solution
        solution, _, _, _ = np.linalg.lstsq(X, y.reshape(-1), rcond=None)
        self.weights = solution[:-1]
        self.bias = solution[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(len(x), -1)
        return x_flat @ self.weights + self.bias


class PersistenceBenchmark(AnalyticalBenchmark):
    """Naive persistence benchmark (predict last value)."""

    def __init__(self):
        super().__init__('persistence')

    def predict(self, x: np.ndarray) -> np.ndarray:
        # For time series: predict last value in sequence
        if x.ndim > 1:
            return x[:, -1] if x.ndim == 2 else x[:, -1, :]
        return x


class MovingAverageBenchmark(AnalyticalBenchmark):
    """Moving average benchmark."""

    def __init__(self, window: int = 10):
        super().__init__(f'moving_avg_{window}')
        self.window = window

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([x[-self.window:].mean()])
        elif x.ndim == 2:
            return x[:, -self.window:].mean(axis=1)
        else:
            return x[:, -self.window:, :].mean(axis=1)


class BlindEvaluator:
    """
    Main evaluator ensuring blind test evaluation.

    Key principle: Test data is NEVER used during training or model selection.
    Only evaluated once at the very end.
    """

    def __init__(
        self,
        experiment_dir: Path,
        benchmarks: Optional[List[AnalyticalBenchmark]] = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.benchmarks = benchmarks or [
            LinearBenchmark(),
            PersistenceBenchmark(),
            MovingAverageBenchmark(5),
            MovingAverageBenchmark(10)
        ]

        self.results: List[EvaluationResult] = []
        self.comparisons: List[BenchmarkComparison] = []

        # Track if blind test has been run (should only happen once)
        self._blind_test_completed = False

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        split: str,
        model_name: str,
        dataset_name: str,
        device: torch.device
    ) -> EvaluationResult:
        """
        Evaluate model on given split.

        Args:
            model: PyTorch model
            dataloader: Data loader for the split
            split: 'train', 'val', or 'test'
            model_name: Name for tracking
            dataset_name: Dataset identifier
            device: Compute device

        Returns:
            EvaluationResult with all metrics
        """
        if split == 'test':
            if self._blind_test_completed:
                raise RuntimeError(
                    "BLIND TEST ALREADY COMPLETED. "
                    "Test set should only be evaluated ONCE at the end."
                )

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)

                # Handle dict output from our models
                if isinstance(output, dict):
                    pred = output['prediction']
                else:
                    pred = output

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = MetricCalculator.compute_all(preds, targets)

        result = EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            split=split,
            metrics=metrics,
            predictions=preds,
            targets=targets
        )

        self.results.append(result)

        if split == 'test':
            self._blind_test_completed = True
            self._save_result(result)

        return result

    def run_benchmarks(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str
    ) -> List[EvaluationResult]:
        """
        Run all analytical benchmarks.

        Fits on training data, evaluates on test.
        """
        results = []

        for benchmark in self.benchmarks:
            # Fit if benchmark supports it
            if hasattr(benchmark, 'fit'):
                benchmark.fit(x_train, y_train)

            result = benchmark.evaluate(x_test, y_test, dataset_name)
            results.append(result)
            self.results.append(result)
            self._save_result(result)

        return results

    def compare_to_benchmarks(
        self,
        model_result: EvaluationResult,
        benchmark_results: List[EvaluationResult],
        alpha: float = 0.05
    ) -> List[BenchmarkComparison]:
        """
        Compare model results to benchmarks with statistical testing.
        """
        comparisons = []

        model_errors = np.abs(model_result.predictions - model_result.targets)

        for bench_result in benchmark_results:
            bench_errors = np.abs(bench_result.predictions - bench_result.targets)

            # Calculate improvement for each metric
            improvement = {}
            p_values = {}
            is_significant = {}

            for metric in model_result.metrics:
                model_val = model_result.metrics[metric]
                bench_val = bench_result.metrics[metric]

                # For most metrics, lower is better (except r2, correlation)
                if metric in ['r2', 'correlation']:
                    improvement[metric] = model_val - bench_val
                else:
                    improvement[metric] = bench_val - model_val  # Positive = model better

            # Statistical test: paired t-test on absolute errors
            t_stat, p_val = stats.ttest_rel(bench_errors.flatten(), model_errors.flatten())
            p_values['error'] = float(p_val)
            is_significant['error'] = p_val < alpha and t_stat > 0  # Model has lower errors

            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_pval = stats.wilcoxon(
                    bench_errors.flatten() - model_errors.flatten(),
                    alternative='greater'
                )
                p_values['wilcoxon'] = float(w_pval)
                is_significant['wilcoxon'] = w_pval < alpha
            except ValueError:
                # Can fail if all differences are zero
                p_values['wilcoxon'] = 1.0
                is_significant['wilcoxon'] = False

            comparison = BenchmarkComparison(
                model_result=model_result,
                benchmark_result=bench_result,
                improvement=improvement,
                p_values=p_values,
                is_significant=is_significant
            )
            comparisons.append(comparison)
            self.comparisons.append(comparison)

        self._save_comparisons(comparisons)
        return comparisons

    def _save_result(self, result: EvaluationResult):
        """Save evaluation result to disk."""
        results_dir = self.experiment_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        filename = f"{result.model_name}_{result.dataset_name}_{result.split}.json"
        with open(results_dir / filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Also save predictions/targets for later analysis
        if result.predictions is not None:
            np.save(
                results_dir / f"{result.model_name}_{result.dataset_name}_predictions.npy",
                result.predictions
            )
            np.save(
                results_dir / f"{result.model_name}_{result.dataset_name}_targets.npy",
                result.targets
            )

    def _save_comparisons(self, comparisons: List[BenchmarkComparison]):
        """Save benchmark comparisons."""
        comp_dir = self.experiment_dir / 'comparisons'
        comp_dir.mkdir(exist_ok=True)

        summary = []
        for comp in comparisons:
            summary.append({
                'model': comp.model_result.model_name,
                'benchmark': comp.benchmark_result.model_name,
                'dataset': comp.model_result.dataset_name,
                'improvement': comp.improvement,
                'p_values': comp.p_values,
                'is_significant': comp.is_significant
            })

        with open(comp_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 60,
            "BLIND EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]

        # Group results by dataset
        datasets = set(r.dataset_name for r in self.results)

        for dataset in datasets:
            lines.append(f"\nDataset: {dataset}")
            lines.append("-" * 40)

            dataset_results = [r for r in self.results if r.dataset_name == dataset]

            # Show test results
            test_results = [r for r in dataset_results if r.split == 'test']
            for result in test_results:
                lines.append(f"\n  {result.model_name}:")
                for metric, value in result.metrics.items():
                    lines.append(f"    {metric}: {value:.6f}")

        # Show comparisons
        if self.comparisons:
            lines.append("\n" + "=" * 60)
            lines.append("BENCHMARK COMPARISONS")
            lines.append("=" * 60)

            for comp in self.comparisons:
                lines.append(f"\n{comp.model_result.model_name} vs {comp.benchmark_result.model_name}:")

                for metric, imp in comp.improvement.items():
                    direction = "↑" if imp > 0 else "↓" if imp < 0 else "="
                    lines.append(f"  {metric}: {direction} {abs(imp):.6f}")

                sig = comp.is_significant.get('error', False)
                p = comp.p_values.get('error', 1.0)
                lines.append(f"  Statistical significance: {'YES' if sig else 'NO'} (p={p:.4f})")

        report = "\n".join(lines)

        # Save report
        with open(self.experiment_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)

        return report
