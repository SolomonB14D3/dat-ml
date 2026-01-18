"""
DAT Pre-training Data Loader.

Loads simulation data from DAT experiments where E6/H3 patterns are explicit.
Used to teach the model what these patterns look like before transfer learning.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union


class DATSimulationDataset(Dataset):
    """
    Dataset from DAT simulation JSON files.

    Each frame contains:
    - pos: 3D physical position
    - vorticity: scalar physical observable
    - offset_6d: 6D E6 lattice offset

    Pre-training tasks:
    1. offset_6d → vorticity (learn E6 to observable mapping)
    2. sequence of states → next state (learn dynamics)
    """

    def __init__(
        self,
        json_paths: Union[str, Path, List[Union[str, Path]]],
        task: str = 'predict_vorticity',  # 'predict_vorticity' or 'predict_dynamics'
        sequence_length: int = 16,
        normalize: bool = True
    ):
        self.task = task
        self.sequence_length = sequence_length
        self.normalize = normalize

        # Load data from all paths
        if isinstance(json_paths, (str, Path)):
            json_paths = [json_paths]

        self.frames = []
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
            self.frames.extend(data)

        # Extract arrays
        self.pos = np.array([f['pos'] for f in self.frames], dtype=np.float32)
        self.vorticity = np.array([f['vorticity'] for f in self.frames], dtype=np.float32)
        self.offset_6d = np.array([f['offset_6d'] for f in self.frames], dtype=np.float32)

        # Normalize
        if normalize:
            self.pos_mean, self.pos_std = self.pos.mean(0), self.pos.std(0) + 1e-8
            self.vort_mean, self.vort_std = self.vorticity.mean(), self.vorticity.std() + 1e-8
            self.off_mean, self.off_std = self.offset_6d.mean(0), self.offset_6d.std(0) + 1e-8

            self.pos = (self.pos - self.pos_mean) / self.pos_std
            self.vorticity = (self.vorticity - self.vort_mean) / self.vort_std
            self.offset_6d = (self.offset_6d - self.off_mean) / self.off_std

        # Compute number of samples based on task
        if task == 'predict_vorticity':
            self.n_samples = len(self.frames)
        elif task == 'predict_dynamics':
            self.n_samples = len(self.frames) - sequence_length
        else:
            raise ValueError(f"Unknown task: {task}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task == 'predict_vorticity':
            # Input: offset_6d (and optionally pos)
            # Output: vorticity
            x = np.concatenate([self.offset_6d[idx], self.pos[idx]])  # 9D input
            y = np.array([self.vorticity[idx]])
            return torch.from_numpy(x), torch.from_numpy(y)

        elif self.task == 'predict_dynamics':
            # Input: sequence of (offset_6d, pos, vorticity)
            # Output: next (offset_6d, pos, vorticity)
            seq_start = idx
            seq_end = idx + self.sequence_length

            # Build input sequence
            x_off = self.offset_6d[seq_start:seq_end]  # (seq, 6)
            x_pos = self.pos[seq_start:seq_end]        # (seq, 3)
            x_vort = self.vorticity[seq_start:seq_end, np.newaxis]  # (seq, 1)
            x = np.concatenate([x_off, x_pos, x_vort], axis=1)  # (seq, 10)

            # Target: next state
            y_off = self.offset_6d[seq_end]
            y_pos = self.pos[seq_end]
            y_vort = np.array([self.vorticity[seq_end]])
            y = np.concatenate([y_off, y_pos, y_vort])  # (10,)

            return torch.from_numpy(x), torch.from_numpy(y)

    def get_input_dim(self) -> int:
        if self.task == 'predict_vorticity':
            return 9  # 6D offset + 3D pos
        else:
            return 10  # 6D offset + 3D pos + 1 vorticity

    def get_output_dim(self) -> int:
        if self.task == 'predict_vorticity':
            return 1
        else:
            return 10


class DATMultiTaskDataset(Dataset):
    """
    Multi-task pre-training dataset that combines multiple prediction tasks.

    Teaches the model:
    1. E6 offset → physical observable
    2. Dynamics prediction
    3. Reconstruction (autoencoder-style)
    """

    def __init__(
        self,
        json_paths: Union[str, Path, List[Union[str, Path]]],
        sequence_length: int = 16,
        normalize: bool = True
    ):
        self.sequence_length = sequence_length

        # Load data
        if isinstance(json_paths, (str, Path)):
            json_paths = [json_paths]

        frames = []
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
            frames.extend(data)

        # Extract arrays
        self.pos = np.array([f['pos'] for f in frames], dtype=np.float32)
        self.vorticity = np.array([f['vorticity'] for f in frames], dtype=np.float32)
        self.offset_6d = np.array([f['offset_6d'] for f in frames], dtype=np.float32)

        # Normalize
        if normalize:
            self.pos = (self.pos - self.pos.mean(0)) / (self.pos.std(0) + 1e-8)
            self.vorticity = (self.vorticity - self.vorticity.mean()) / (self.vorticity.std() + 1e-8)
            self.offset_6d = (self.offset_6d - self.offset_6d.mean(0)) / (self.offset_6d.std(0) + 1e-8)

        self.n_frames = len(frames)
        # Each valid sequence can generate multiple tasks
        self.n_sequences = self.n_frames - sequence_length

    def __len__(self) -> int:
        # 3 tasks per sequence position
        return self.n_sequences * 3

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Decode task and position
        seq_idx = idx // 3
        task_idx = idx % 3

        seq_start = seq_idx
        seq_end = seq_idx + self.sequence_length

        if task_idx == 0:
            # Task 0: Predict vorticity from offset_6d
            x = np.concatenate([self.offset_6d[seq_end], self.pos[seq_end]])
            y = np.array([self.vorticity[seq_end]])

        elif task_idx == 1:
            # Task 1: Predict next offset_6d from sequence
            x = self.offset_6d[seq_start:seq_end].flatten()
            y = self.offset_6d[seq_end]

        else:
            # Task 2: Predict next full state from sequence
            x_seq = np.concatenate([
                self.offset_6d[seq_start:seq_end],
                self.pos[seq_start:seq_end],
                self.vorticity[seq_start:seq_end, np.newaxis]
            ], axis=1).flatten()
            y = np.concatenate([
                self.offset_6d[seq_end],
                self.pos[seq_end],
                [self.vorticity[seq_end]]
            ])
            x = x_seq

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), task_idx


def load_dat_pretrain_data(
    gpu_discovery_path: str = '/Users/bryan/dat_gpu_discovery.json',
    infinite_scaling_path: str = '/Users/bryan/dat_infinite_scaling.json',
    task: str = 'predict_vorticity',
    sequence_length: int = 16
) -> DATSimulationDataset:
    """
    Load combined DAT pre-training data.
    """
    return DATSimulationDataset(
        json_paths=[gpu_discovery_path, infinite_scaling_path],
        task=task,
        sequence_length=sequence_length,
        normalize=True
    )
