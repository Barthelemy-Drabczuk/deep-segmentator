"""Training-specific configurations."""
from pydantic import BaseModel

from .base_config import TrainingConfig, load_config

__all__ = ["TrainingConfig", "load_config", "WarmupCosineSchedulerConfig"]


class WarmupCosineSchedulerConfig(BaseModel):
    """Configuration for the warmup cosine annealing learning rate scheduler."""

    warmup_epochs: int = 5
    total_epochs: int = 200
    eta_min: float = 1e-6
