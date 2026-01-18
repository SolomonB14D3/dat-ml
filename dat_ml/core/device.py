"""
Device configuration optimized for Apple M3 Ultra.
- 28 cores (20 performance + 8 efficiency)
- 96GB unified memory
- MPS (Metal Performance Shaders) backend
"""

import torch
import os
from dataclasses import dataclass
from typing import Optional

# Optimize for M3 Ultra
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Allow full memory usage
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for unsupported ops (e.g., cdist_backward)


@dataclass
class HardwareConfig:
    """Hardware configuration for M3 Ultra Mac Studio."""
    total_cores: int = 28
    performance_cores: int = 20
    efficiency_cores: int = 8
    memory_gb: int = 96
    # Reserve ~10GB for system, use rest for training
    max_memory_gb: int = 86


def get_device() -> torch.device:
    """Get optimal device for M3 Ultra."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimal_workers() -> int:
    """Get optimal DataLoader workers for M3 Ultra."""
    # Use efficiency cores for data loading, leave performance cores for compute
    return 8


def get_optimal_batch_size(model_memory_mb: float, sample_memory_mb: float) -> int:
    """
    Calculate optimal batch size given model and sample memory requirements.
    Targets ~70% of available unified memory for training headroom.
    """
    config = HardwareConfig()
    available_mb = config.max_memory_gb * 1024 * 0.7
    usable_mb = available_mb - model_memory_mb
    batch_size = int(usable_mb / sample_memory_mb)
    # Round down to power of 2 for efficiency
    return 2 ** (batch_size.bit_length() - 1) if batch_size > 0 else 1


def configure_torch_threads():
    """Configure PyTorch threading for M3 Ultra."""
    config = HardwareConfig()
    # Use performance cores for compute
    torch.set_num_threads(config.performance_cores)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(config.efficiency_cores)


# Auto-configure on import
configure_torch_threads()
DEVICE = get_device()
HARDWARE = HardwareConfig()
