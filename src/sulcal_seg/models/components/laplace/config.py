"""Configuration for the Laplace geometry component."""
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from sulcal_seg.config.base_config import BaseConfig


class LaplaceComponentConfig(BaseConfig):
    """Configuration for the Laplace-Beltrami smoothing component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Smoothing
    num_iterations: int = Field(default=30, ge=1)
    lambda_smooth: float = Field(default=0.5, gt=0.0, le=1.0)
    algorithm: Literal["taubin", "laplacian"] = "taubin"

    # Mesh extraction
    iso_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    smooth_before_mesh: bool = True
    marching_cubes_level: float = Field(default=0.5, gt=0.0)

    # Topology validation
    target_genus: int = Field(default=0, ge=0)
    max_holes: int = Field(default=0, ge=0)

    # Rasterisation
    voxel_size_mm: float = Field(default=1.0, gt=0.0)

    REQUIRED_KEYS = frozenset({"num_iterations", "lambda_smooth"})

    @field_validator("lambda_smooth")
    @classmethod
    def _valid_lambda(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"lambda_smooth must be in (0, 1], got {v}")
        return v
