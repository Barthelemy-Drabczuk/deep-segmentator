"""Ablation study configuration."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AblationStudyConfig(BaseModel):
    """Configuration for a single ablation study variant."""

    name: str
    description: str
    eas_enabled: bool = False
    gan_enabled: bool = False
    laplace_enabled: bool = False
    dual_label: bool = True
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


# Pre-defined ablation studies matching COMPLETE_MODEL_SPECIFICATION
ABLATION_STUDIES = [
    AblationStudyConfig(
        name="ablation_1_baseline",
        description="Hand-designed 3D U-Net baseline (no EAS, no GAN, no Laplace)",
        eas_enabled=False, gan_enabled=False, laplace_enabled=False,
    ),
    AblationStudyConfig(
        name="ablation_2_nas_only",
        description="EAS-discovered architecture, Dice loss only",
        eas_enabled=True, gan_enabled=False, laplace_enabled=False,
    ),
    AblationStudyConfig(
        name="ablation_3_nas_gan_single",
        description="EAS + single-label GAN (Morphologist only)",
        eas_enabled=True, gan_enabled=True, laplace_enabled=False, dual_label=False,
    ),
    AblationStudyConfig(
        name="ablation_4_nas_laplace",
        description="EAS + Laplace constraint (no GAN)",
        eas_enabled=True, gan_enabled=False, laplace_enabled=True,
    ),
    AblationStudyConfig(
        name="ablation_5a_full_dual_label",
        description="Full method: EAS + dual-label GAN + Laplace (PROPOSED METHOD)",
        eas_enabled=True, gan_enabled=True, laplace_enabled=True, dual_label=True,
    ),
    AblationStudyConfig(
        name="ablation_5b_full_single_label",
        description="Full method with single-label GAN (comparison for mode collapse)",
        eas_enabled=True, gan_enabled=True, laplace_enabled=True, dual_label=False,
    ),
    AblationStudyConfig(
        name="ablation_6_one_stage",
        description="Full method with single-stage training (no warmup phase)",
        eas_enabled=True, gan_enabled=True, laplace_enabled=True,
        config_overrides={"gan": {"warmup_epochs": 0}},
    ),
]


class AblationSuiteConfig(BaseModel):
    """Configuration for a full suite of ablation studies."""

    studies: List[AblationStudyConfig] = Field(default_factory=lambda: ABLATION_STUDIES)
    base_config_path: str = "configs/default.yaml"
    output_dir: str = "results/ablations"
