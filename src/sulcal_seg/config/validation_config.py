"""Validation and evaluation configuration."""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """Configuration for model validation and evaluation."""

    metrics: List[str] = Field(
        default_factory=lambda: [
            "dice",
            "hausdorff",
            "sulcal_depth",
            "sulcal_width",
            "entropy",
            "fid",
            "ndb",
            "dice_balance",
            "mesh_quality",
            "topology",
            "landmarks",
            "cross_cohort",
            "latency",
        ]
    )
    eval_every_n_epochs: int = 5
    early_stopping_patience: int = 20
    save_best_metric: str = "dice"
    output_dir: Path = Path("results")
    num_sulci_labels: int = 40
    hausdorff_threshold_mm: float = 2.5
    dice_threshold: float = 0.93
    entropy_threshold: float = 0.7
    inference_time_threshold_sec: float = 300.0  # 5 min
