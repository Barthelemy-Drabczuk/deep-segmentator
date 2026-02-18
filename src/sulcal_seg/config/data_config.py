"""Dataset-specific configuration extensions."""
from .base_config import DataConfig

__all__ = ["DataConfig"]


class UKBiobankDataConfig(DataConfig):
    """UK Biobank specific data configuration."""

    dataset_name: str = "ukbiobank"


class ABCDDataConfig(DataConfig):
    """ABCD cohort specific data configuration."""

    dataset_name: str = "abcd"


class ABIDEDataConfig(DataConfig):
    """ABIDE cohort specific data configuration."""

    dataset_name: str = "abide"


class SENIORDataConfig(DataConfig):
    """SENIOR cohort specific data configuration."""

    dataset_name: str = "senior"
