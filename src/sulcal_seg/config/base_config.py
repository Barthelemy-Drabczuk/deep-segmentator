"""
Base configuration classes for sulcal segmentation.
All configs use Pydantic v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""

    dataset_name: str = Field(..., description="Dataset to use (ukbiobank, abcd, abide, senior, custom)")
    root_dir: Path = Field(..., description="Root data directory")
    splits: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    loader_params: Dict[str, Any] = Field(default_factory=dict)


class EASConfig(BaseConfig):
    """Evolutionary Architecture Search configuration."""

    enabled: bool = True
    population_size: int = 30
    generations: int = 100
    search_space: Dict[str, Any] = Field(
        default_factory=lambda: {
            "encoder_depths": [3, 4, 5, 6],
            "filters": [32, 64, 128, 256],
            "conv_types": ["conv3d", "dilated_conv3d"],
            "activations": ["leaky_relu", "relu"],
        }
    )
    hosvd_rank_ratios: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    mutation_rate: float = 0.1
    quick_eval_epochs: int = 5

    @field_validator("hosvd_rank_ratios", mode="before")
    @classmethod
    def validate_ratios(cls, v: Any) -> Tuple[float, float, float]:
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        if len(v) != 3:
            raise ValueError("hosvd_rank_ratios must have exactly 3 elements")
        if not all(0.0 < r <= 1.0 for r in v):
            raise ValueError("All hosvd_rank_ratios must be in (0, 1]")
        return v  # type: ignore[return-value]

    @field_validator("mutation_rate")
    @classmethod
    def validate_mutation_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        return v


class GANConfig(BaseConfig):
    """Multi-label adversarial training configuration."""

    enabled: bool = True
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0004
    batch_size: int = 12
    dual_label: bool = True
    warmup_epochs: int = 5
    total_epochs: int = 200
    dice_weight: float = 100.0
    gan_weight: float = 1.0
    laplace_weight: float = 5.0
    num_sulci_labels: int = 40
    base_filters: int = 32
    encoder_depth: int = 4


class LaplaceConfig(BaseConfig):
    """Laplace geometry constraint configuration."""

    enabled: bool = True
    smoothing_weight: float = 0.5
    iterations: int = 10
    preserve_landmarks: bool = True
    lambda_smooth: float = 0.3
    mu_reweight: float = -0.33
    max_iterations: int = 50


class TrainingConfig(BaseConfig):
    """Master training configuration."""

    data: DataConfig
    eas: EASConfig = Field(default_factory=EASConfig)
    gan: GANConfig = Field(default_factory=GANConfig)
    laplace: LaplaceConfig = Field(default_factory=LaplaceConfig)
    device: str = "cuda"
    num_workers: int = 8
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42


def load_config(yaml_path: Path) -> TrainingConfig:
    """Load TrainingConfig from a YAML file."""
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)
