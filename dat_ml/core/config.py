"""
Central configuration for DAT-ML project.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import json


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    raw_path: Path = DATA_RAW
    processed_path: Path = DATA_PROCESSED
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15  # Blind holdout
    seed: int = 42

    def __post_init__(self):
        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DATConfig:
    """Configuration for Discrete Alignment Theory transforms."""
    # E6 lattice projection parameters
    e6_dimension: int = 6
    projection_dimension: int = 3  # Target H3 manifold
    perpendicular_dimension: int = 4  # 4D perpendicular slice for phason steering

    # Topological parameters
    icosahedral_tau: float = 1.618033988749895  # Golden ratio
    phason_rotation_steps: int = 64

    # Stability thresholds from DAT validation
    localization_target: float = 4.2  # Target improvement factor


@dataclass
class H3HybridConfig:
    """Configuration inspired by H3-Hybrid-Discovery findings."""
    # Core structural parameters
    coordination_peak: float = 1.081  # RDF peak in sigma units
    energy_depth: float = -7.68  # epsilon per atom
    baseline_energy: float = -6.48  # FCC/HCP equilibrium

    # Stability parameters
    annealing_start_temp: float = 0.2
    annealing_end_temp: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration optimized for M3 Ultra."""
    batch_size: int = 256
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10

    # M3 Ultra optimizations
    num_workers: int = 8  # Efficiency cores for data loading
    pin_memory: bool = False  # Not needed with unified memory
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # Gradient accumulation for effective larger batches
    gradient_accumulation_steps: int = 4


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    dat: DATConfig = field(default_factory=DATConfig)
    h3: H3HybridConfig = field(default_factory=H3HybridConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = EXPERIMENTS_DIR / self.name / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
