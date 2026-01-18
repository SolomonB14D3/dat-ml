"""
Multi-format data ingestion pipeline.
Supports: time series, physical systems, networks, tabular data.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import json
import csv

from ..core.config import DataConfig, TrainingConfig, DATA_RAW
from ..core.device import get_optimal_workers, DEVICE


class BaseDataset(Dataset, ABC):
    """Abstract base for all dataset types."""

    def __init__(self, data_path: Path, transform: Optional[Callable] = None):
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self) -> List:
        """Load raw data from path."""
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TimeSeriesDataset(BaseDataset):
    """Dataset for time series prediction tasks."""

    def __init__(
        self,
        data_path: Path,
        sequence_length: int = 64,
        horizon: int = 1,
        transform: Optional[Callable] = None
    ):
        self.sequence_length = sequence_length
        self.horizon = horizon
        super().__init__(data_path, transform)

    def _load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load and window time series data."""
        # Support multiple formats
        if self.data_path.suffix == '.npy':
            raw = np.load(self.data_path)
        elif self.data_path.suffix == '.csv':
            raw = np.loadtxt(self.data_path, delimiter=',', skiprows=1)
        elif self.data_path.suffix == '.json':
            with open(self.data_path) as f:
                raw = np.array(json.load(f)['values'])
        else:
            raise ValueError(f"Unsupported format: {self.data_path.suffix}")

        # Ensure 2D: (time, features)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        # Create sliding windows
        samples = []
        for i in range(len(raw) - self.sequence_length - self.horizon + 1):
            x = raw[i:i + self.sequence_length]
            y = raw[i + self.sequence_length:i + self.sequence_length + self.horizon]
            samples.append((x, y.flatten()))

        return samples


class PhysicalSystemDataset(BaseDataset):
    """Dataset for physical system simulations (MD, particle systems)."""

    def __init__(
        self,
        data_path: Path,
        transform: Optional[Callable] = None
    ):
        super().__init__(data_path, transform)

    def _load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load physical system trajectories."""
        if self.data_path.is_dir():
            # Load LAMMPS-style dump files or trajectory folders
            samples = self._load_trajectory_dir()
        elif self.data_path.suffix == '.npy':
            data = np.load(self.data_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                samples = [(d['state'], d['target']) for d in data]
            else:
                # Assume consecutive frames: predict next from current
                samples = [(data[i], data[i + 1]) for i in range(len(data) - 1)]
        else:
            raise ValueError(f"Unsupported format: {self.data_path}")

        return samples

    def _load_trajectory_dir(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load trajectory from directory of frame files."""
        frames = sorted(self.data_path.glob("*.npy"))
        samples = []
        for i in range(len(frames) - 1):
            current = np.load(frames[i])
            next_frame = np.load(frames[i + 1])
            samples.append((current.flatten(), next_frame.flatten()))
        return samples


class NetworkDataset(BaseDataset):
    """Dataset for graph/network data."""

    def __init__(
        self,
        data_path: Path,
        transform: Optional[Callable] = None
    ):
        super().__init__(data_path, transform)

    def _load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load network data as adjacency + features."""
        if self.data_path.suffix == '.json':
            with open(self.data_path) as f:
                graphs = json.load(f)
            samples = []
            for g in graphs:
                adj = np.array(g['adjacency'])
                features = np.array(g.get('features', np.eye(adj.shape[0])))
                target = np.array(g['target'])
                # Flatten adjacency + features as input
                x = np.concatenate([adj.flatten(), features.flatten()])
                samples.append((x, target))
            return samples
        elif self.data_path.suffix == '.npz':
            data = np.load(self.data_path)
            return list(zip(data['inputs'], data['targets']))
        else:
            raise ValueError(f"Unsupported network format: {self.data_path.suffix}")


class TabularDataset(BaseDataset):
    """Generic tabular dataset."""

    def __init__(
        self,
        data_path: Path,
        target_col: Union[int, str] = -1,
        transform: Optional[Callable] = None
    ):
        self.target_col = target_col
        super().__init__(data_path, transform)

    def _load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load tabular data."""
        if self.data_path.suffix == '.csv':
            with open(self.data_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                data = [row for row in reader]

            data = np.array(data, dtype=np.float32)

            if isinstance(self.target_col, str):
                target_idx = header.index(self.target_col)
            else:
                target_idx = self.target_col

            targets = data[:, target_idx]
            features = np.delete(data, target_idx, axis=1)
            return [(features[i], targets[i:i+1]) for i in range(len(data))]

        elif self.data_path.suffix == '.npy':
            data = np.load(self.data_path)
            return [(data[i, :-1], data[i, -1:]) for i in range(len(data))]

        raise ValueError(f"Unsupported format: {self.data_path.suffix}")


def create_dataloaders(
    dataset: Dataset,
    config: DataConfig,
    training_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset and create train/val/test dataloaders.
    Test set is the BLIND holdout - never used during training.
    """
    n = len(dataset)
    train_n = int(n * config.train_ratio)
    val_n = int(n * config.val_ratio)
    test_n = n - train_n - val_n

    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_n, val_n, test_n], generator=generator
    )

    # Check if dataset has pre-loaded GPU data (can't use multiprocessing)
    has_gpu_data = hasattr(dataset, 'data_tensor') and dataset.data_tensor is not None

    if has_gpu_data:
        # Data pre-loaded to GPU - must use single process
        loader_kwargs = {
            'batch_size': training_config.batch_size,
            'num_workers': 0,  # GPU data can't be shared across processes
            'pin_memory': False,
        }
    else:
        loader_kwargs = {
            'batch_size': training_config.batch_size,
            'num_workers': training_config.num_workers,
            'pin_memory': training_config.pin_memory,
            'persistent_workers': training_config.persistent_workers if training_config.num_workers > 0 else False,
            'prefetch_factor': training_config.prefetch_factor if training_config.num_workers > 0 else None,
        }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    # Test loader - BLIND HOLDOUT
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def auto_detect_dataset(data_path: Path, **kwargs) -> BaseDataset:
    """Auto-detect dataset type from file structure/content."""
    if data_path.is_dir():
        # Check for trajectory-style data
        if any(data_path.glob("*.npy")) or any(data_path.glob("*.dump")):
            return PhysicalSystemDataset(data_path, **kwargs)

    if data_path.suffix == '.json':
        with open(data_path) as f:
            sample = json.load(f)
        if isinstance(sample, list) and 'adjacency' in sample[0]:
            return NetworkDataset(data_path, **kwargs)
        if 'values' in sample:
            return TimeSeriesDataset(data_path, **kwargs)

    if data_path.suffix == '.csv':
        with open(data_path) as f:
            header = f.readline().lower()
        if 'time' in header or 'date' in header:
            return TimeSeriesDataset(data_path, **kwargs)
        return TabularDataset(data_path, **kwargs)

    if data_path.suffix == '.npy':
        data = np.load(data_path, allow_pickle=True)
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] < 10):
            return TimeSeriesDataset(data_path, **kwargs)
        return PhysicalSystemDataset(data_path, **kwargs)

    raise ValueError(f"Could not auto-detect dataset type for {data_path}")
