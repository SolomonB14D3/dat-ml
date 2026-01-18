"""
NetCDF Data Loader for Atmospheric/Climate Data.

Handles ERA5 and similar gridded datasets for prediction tasks.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class SpatialSubset:
    """Define a spatial region to extract."""
    lat_min: float = -90
    lat_max: float = 90
    lon_min: float = 0
    lon_max: float = 360
    coarsen_factor: int = 1  # Spatial downsampling


class ERA5Dataset(Dataset):
    """
    Dataset for ERA5 atmospheric prediction.

    Prediction task: Given past N days, predict future M days.
    """

    def __init__(
        self,
        nc_path: Union[str, Path],
        variable: str = 'z',
        sequence_length: int = 14,      # Past days to use as input
        horizon: int = 1,                # Days ahead to predict
        spatial_subset: Optional[SpatialSubset] = None,
        normalize: bool = True,
        flatten_spatial: bool = True
    ):
        self.nc_path = Path(nc_path)
        self.variable = variable
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalize = normalize
        self.flatten_spatial = flatten_spatial

        # Load data
        self.ds = xr.open_dataset(self.nc_path)
        self.data = self._extract_data(spatial_subset)

        # Normalize
        if normalize:
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0
            self.std = 1

        # Number of valid samples
        self.n_samples = len(self.data) - sequence_length - horizon + 1

    def _extract_data(self, subset: Optional[SpatialSubset]) -> np.ndarray:
        """Extract and optionally subset the data."""
        data = self.ds[self.variable].values

        # Remove pressure_level dimension if singleton
        if data.ndim == 4 and data.shape[1] == 1:
            data = data.squeeze(axis=1)  # Now (time, lat, lon)

        if subset is not None:
            # Get lat/lon indices
            lats = self.ds.latitude.values
            lons = self.ds.longitude.values

            lat_mask = (lats >= subset.lat_min) & (lats <= subset.lat_max)
            lon_mask = (lons >= subset.lon_min) & (lons <= subset.lon_max)

            data = data[:, lat_mask, :][:, :, lon_mask]

            # Coarsen if requested
            if subset.coarsen_factor > 1:
                cf = subset.coarsen_factor
                data = data[:, ::cf, ::cf]

        return data.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: past sequence_length days
        x = self.data[idx:idx + self.sequence_length]

        # Target: horizon days ahead
        target_idx = idx + self.sequence_length + self.horizon - 1
        y = self.data[target_idx]

        if self.flatten_spatial:
            # Flatten spatial dims: (seq, lat, lon) -> (seq, lat*lon)
            x = x.reshape(self.sequence_length, -1)
            y = y.flatten()

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_input_dim(self) -> int:
        """Get flattened input dimension."""
        if self.flatten_spatial:
            return self.data.shape[1] * self.data.shape[2]
        return self.data.shape[1:]

    def close(self):
        """Close the underlying NetCDF file."""
        self.ds.close()


class ERA5PatchDataset(Dataset):
    """
    Patch-based dataset for local predictions.

    Extracts spatial patches for more manageable input sizes.
    Good for testing if DAT/H3 helps with local dynamics.
    """

    def __init__(
        self,
        nc_path: Union[str, Path],
        variable: str = 'z',
        sequence_length: int = 14,
        horizon: int = 1,
        patch_size: int = 32,           # Spatial patch size
        stride: int = 16,                # Stride between patches
        normalize: bool = True,
        max_patches_per_time: int = 100  # Limit patches per timestep
    ):
        self.nc_path = Path(nc_path)
        self.variable = variable
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize

        # Load data
        self.ds = xr.open_dataset(self.nc_path)
        data = self.ds[variable].values

        # Remove singleton dimensions
        if data.ndim == 4 and data.shape[1] == 1:
            data = data.squeeze(axis=1)

        self.data = data.astype(np.float32)

        # Normalize
        if normalize:
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / (self.std + 1e-8)

        # Compute patch locations
        self.patch_locs = self._compute_patch_locations(max_patches_per_time)

        # Number of valid time windows
        self.n_time_windows = len(self.data) - sequence_length - horizon + 1

        # Total samples
        self.n_samples = self.n_time_windows * len(self.patch_locs)

    def _compute_patch_locations(self, max_patches: int) -> List[Tuple[int, int]]:
        """Compute valid patch starting locations."""
        _, n_lat, n_lon = self.data.shape

        locs = []
        for i in range(0, n_lat - self.patch_size + 1, self.stride):
            for j in range(0, n_lon - self.patch_size + 1, self.stride):
                locs.append((i, j))

        # Subsample if too many
        if len(locs) > max_patches:
            indices = np.linspace(0, len(locs) - 1, max_patches, dtype=int)
            locs = [locs[i] for i in indices]

        return locs

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Decode index
        time_idx = idx // len(self.patch_locs)
        patch_idx = idx % len(self.patch_locs)
        lat_start, lon_start = self.patch_locs[patch_idx]

        # Extract patch sequence
        x = self.data[
            time_idx:time_idx + self.sequence_length,
            lat_start:lat_start + self.patch_size,
            lon_start:lon_start + self.patch_size
        ]

        # Target patch
        target_time = time_idx + self.sequence_length + self.horizon - 1
        y = self.data[
            target_time,
            lat_start:lat_start + self.patch_size,
            lon_start:lon_start + self.patch_size
        ]

        # Flatten: (seq, patch, patch) -> (seq, patch*patch)
        x = x.reshape(self.sequence_length, -1)
        y = y.flatten()

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_input_dim(self) -> int:
        """Get flattened patch dimension."""
        return self.patch_size * self.patch_size

    def close(self):
        self.ds.close()


class ERA5RegionalDataset(Dataset):
    """
    Regional average dataset - simplest form for initial testing.

    Predicts regional mean Z500 from past regional means.
    Reduces ~1M grid points to a handful of regional indices.
    """

    def __init__(
        self,
        nc_path: Union[str, Path],
        variable: str = 'z',
        sequence_length: int = 30,
        horizon: int = 7,
        regions: Optional[dict] = None,
        normalize: bool = True,
        preload_device: Optional[str] = None  # 'mps', 'cuda', or None for CPU
    ):
        self.nc_path = Path(nc_path)
        self.variable = variable
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalize = normalize

        # Default regions: major atmospheric action centers
        if regions is None:
            regions = {
                'north_atlantic': {'lat': (30, 60), 'lon': (280, 360)},
                'north_pacific': {'lat': (30, 60), 'lon': (150, 220)},
                'europe': {'lat': (35, 70), 'lon': (0, 40)},
                'asia': {'lat': (30, 60), 'lon': (80, 140)},
                'arctic': {'lat': (70, 90), 'lon': (0, 360)},
                'tropics': {'lat': (-20, 20), 'lon': (0, 360)},
            }
        self.regions = regions

        # Load and compute regional means
        self.ds = xr.open_dataset(self.nc_path)
        self.data = self._compute_regional_means()

        # Normalize
        if normalize:
            self.mean = self.data.mean(axis=0, keepdims=True)
            self.std = self.data.std(axis=0, keepdims=True)
            self.data = (self.data - self.mean) / (self.std + 1e-8)

        self.n_samples = len(self.data) - sequence_length - horizon + 1

        # Pre-load to GPU memory if requested
        self.preload_device = preload_device
        if preload_device is not None:
            print(f"Pre-loading ERA5 data to {preload_device}...")
            self.data_tensor = torch.from_numpy(self.data).to(preload_device)
            print(f"  Data shape: {self.data_tensor.shape}, device: {self.data_tensor.device}")
        else:
            self.data_tensor = None

    def _compute_regional_means(self) -> np.ndarray:
        """Compute mean Z500 for each region at each timestep."""
        data = self.ds[self.variable]

        # Remove pressure level if present
        if 'pressure_level' in data.dims:
            data = data.isel(pressure_level=0)

        lats = self.ds.latitude.values
        lons = self.ds.longitude.values

        regional_means = []
        for name, bounds in self.regions.items():
            lat_min, lat_max = bounds['lat']
            lon_min, lon_max = bounds['lon']

            lat_mask = (lats >= lat_min) & (lats <= lat_max)

            # Handle longitude wrap-around
            if lon_min < lon_max:
                lon_mask = (lons >= lon_min) & (lons <= lon_max)
            else:
                lon_mask = (lons >= lon_min) | (lons <= lon_max)

            # Weight by cos(latitude) for proper area averaging
            weights = np.cos(np.radians(lats[lat_mask]))
            weights = weights / weights.sum()

            region_data = data.values[:, lat_mask, :][:, :, lon_mask]

            # Weighted mean over lat, simple mean over lon
            region_mean = np.average(
                region_data.mean(axis=2),
                axis=1,
                weights=weights
            )
            regional_means.append(region_mean)

        return np.stack(regional_means, axis=1).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_idx = idx + self.sequence_length + self.horizon - 1

        if self.data_tensor is not None:
            # Use pre-loaded GPU tensor
            x = self.data_tensor[idx:idx + self.sequence_length]
            y = self.data_tensor[target_idx]
            return x, y
        else:
            # Load from numpy
            x = self.data[idx:idx + self.sequence_length]
            y = self.data[target_idx]
            return torch.from_numpy(x), torch.from_numpy(y)

    def get_input_dim(self) -> int:
        return len(self.regions)

    def get_region_names(self) -> List[str]:
        return list(self.regions.keys())

    def close(self):
        self.ds.close()
