"""
Base configuration classes for sulcal segmentation.
All configs use Pydantic v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""

    dataset_name: str = Field(..., description="Dataset to use (hcp, ukbiobank, abcd, abide, senior, custom)")
    root_dir: Path = Field(..., description="Root data directory")
    splits: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    loader_params: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseConfig):
    """Master training configuration."""

    data: DataConfig
    device: str = "cuda"
    num_workers: int = 8
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42


def load_config(yaml_path: Path) -> TrainingConfig:
    """Load TrainingConfig from a YAML file."""
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)
