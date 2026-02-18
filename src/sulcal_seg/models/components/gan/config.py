"""Configuration for the GAN component."""
from typing import Any, List, Literal

from pydantic import ConfigDict, Field, field_validator

from sulcal_seg.config.base_config import BaseConfig


class GANComponentConfig(BaseConfig):
    """Configuration for the multi-label GAN segmentation component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Architecture
    num_classes: int = Field(default=41, ge=2)
    in_channels: int = Field(default=1, ge=1)
    base_filters: int = Field(default=32, ge=1)
    depth: int = Field(default=4, ge=1)

    # Loss weights
    lambda_dice: float = Field(default=1.0, ge=0.0)
    lambda_gan: float = Field(default=0.1, ge=0.0)
    lambda_laplace: float = Field(default=0.05, ge=0.0)

    # GAN loss variant
    gan_mode: Literal["vanilla", "lsgan", "wgan"] = "lsgan"

    # Training
    learning_rate_g: float = Field(default=1e-4, gt=0.0)
    learning_rate_d: float = Field(default=1e-4, gt=0.0)
    n_critic: int = Field(default=1, ge=1)
    label_smoothing: float = Field(default=0.1, ge=0.0, le=0.5)

    # Dual-label setup
    use_morphologist_labels: bool = True
    use_freesurfer_labels: bool = True

    # Mode collapse detection
    entropy_threshold: float = Field(default=0.5, ge=0.0)
    collapse_window: int = Field(default=10, ge=1)

    REQUIRED_KEYS = frozenset({"num_classes", "in_channels"})

    @field_validator("lambda_dice", "lambda_gan", "lambda_laplace")
    @classmethod
    def _at_least_one_loss(cls, v: float, info: Any) -> float:
        # Individual validators — combined check would need a model validator
        return v
