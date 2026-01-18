"""
DAT-ML: Discrete Alignment Theory Machine Learning Framework

A predictive modeling framework based on principles from:
- Discrete Alignment Theory (DAT) - E6 lattice projections, H3 manifold
- H3-Hybrid-Discovery - Metastable phase compression, energy landscapes
"""

from .core.device import DEVICE, HARDWARE
from .core.config import ExperimentConfig, DataConfig, TrainingConfig

from .models.h3_network import DAT_H3_Predictor, EnsemblePredictor
from .models.baseline import StandardMLP, StandardTransformer, MatchedBaseline
from .transforms.e6_projection import DATTransform, E6Embedding, H3Projection
from .transforms.h3_hybrid import H3HybridTransform, CompressionTransform
from .spectral_dat_layer import (
    SpectralDATLayer, SpectralDATBlock, SpectralDATPredictor,
    TAU, DELTA_0, H3_COORDINATION, OPTIMAL_SIGMA
)

from .data.loader import auto_detect_dataset, create_dataloaders
from .losses.topological import CombinedTopologicalLoss
from .evaluation.blind_eval import BlindEvaluator

from .train import run_experiment, Trainer
from .compare import run_comparison, ABComparison

__version__ = '0.1.0'
__all__ = [
    'DEVICE', 'HARDWARE',
    'ExperimentConfig', 'DataConfig', 'TrainingConfig',
    'DAT_H3_Predictor', 'EnsemblePredictor',
    'StandardMLP', 'StandardTransformer', 'MatchedBaseline',
    'DATTransform', 'E6Embedding', 'H3Projection',
    'H3HybridTransform', 'CompressionTransform',
    'auto_detect_dataset', 'create_dataloaders',
    'CombinedTopologicalLoss',
    'BlindEvaluator',
    'run_experiment', 'Trainer',
    'run_comparison', 'ABComparison'
]
