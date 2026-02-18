"""GPU/CPU device management utilities."""
from typing import List

import torch

from .logging import get_logger

_logger = get_logger(__name__)


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Get the best available compute device.

    Args:
        preferred: Preferred device type ('cuda', 'mps', or 'cpu')

    Returns:
        torch.device to use for computation
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        _logger.info("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        if preferred != "cpu":
            _logger.warning(f"Requested device '{preferred}' not available, falling back to CPU")
    return device


def get_available_gpus() -> List[int]:
    """Return list of available GPU indices."""
    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get GPU memory statistics for a specific device."""
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "free": 0}
    total = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    return {
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "free_gb": (total - allocated) / 1e9,
    }
