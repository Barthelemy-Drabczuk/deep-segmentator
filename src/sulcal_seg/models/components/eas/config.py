"""Configuration for the EAS (Evolutionary Architecture Search) component."""
from typing import Any, List, Tuple

from pydantic import ConfigDict, Field, field_validator

from sulcal_seg.config.base_config import BaseConfig


class EASComponentConfig(BaseConfig):
    """Configuration for the EAS evolutionary search component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Population
    population_size: int = Field(default=20, ge=4)
    num_generations: int = Field(default=50, ge=1)
    elite_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    tournament_size: int = Field(default=4, ge=2)

    # HOSVD breeding
    hosvd_rank_ratios: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    hosvd_n_trials: int = Field(default=5, ge=1)

    # Architecture search space bounds
    min_filters: int = Field(default=8, ge=1)
    max_filters: int = Field(default=256, ge=1)
    min_depth: int = Field(default=2, ge=1)
    max_depth: int = Field(default=6, ge=1)
    allowed_activations: List[str] = Field(
        default_factory=lambda: ["relu", "leaky_relu", "elu", "gelu"]
    )

    # Training per individual
    proxy_epochs: int = Field(default=5, ge=1)
    num_classes: int = Field(default=41, ge=2)

    @field_validator("hosvd_rank_ratios", mode="before")
    @classmethod
    def _coerce_to_tuple(cls, v: Any) -> Tuple[float, float, float]:
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        if len(v) != 3:
            raise ValueError("hosvd_rank_ratios must have exactly 3 elements")
        for r in v:
            if not (0.0 < r <= 1.0):
                raise ValueError(f"Each ratio must be in (0, 1], got {r}")
        return v  # type: ignore[return-value]

    @field_validator("max_filters")
    @classmethod
    def _filters_ordering(cls, v: int, info: Any) -> int:
        min_f = info.data.get("min_filters", 8)
        if v < min_f:
            raise ValueError(f"max_filters ({v}) must be >= min_filters ({min_f})")
        return v

    @field_validator("max_depth")
    @classmethod
    def _depth_ordering(cls, v: int, info: Any) -> int:
        min_d = info.data.get("min_depth", 2)
        if v < min_d:
            raise ValueError(f"max_depth ({v}) must be >= min_depth ({min_d})")
        return v

    REQUIRED_KEYS = frozenset({"population_size", "num_generations", "num_classes"})
