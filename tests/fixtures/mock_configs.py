"""Minimal mock config dicts for unit tests (no GPU or data required)."""
from typing import Any, Dict


def make_eas_config(**overrides: Any) -> Dict[str, Any]:
    """
    Minimal valid EAS component config dict.

    Args:
        **overrides: Key-value pairs to override defaults.
    """
    cfg: Dict[str, Any] = {
        "population_size": 4,
        "num_generations": 2,
        "num_classes": 4,
        "min_filters": 8,
        "max_filters": 32,
        "min_depth": 2,
        "max_depth": 3,
        "mutation_rate": 0.3,
        "crossover_rate": 0.7,
        "tournament_size": 2,
        "hosvd_rank_ratios": [0.7, 0.7, 0.7],
        "proxy_epochs": 1,
        "elite_fraction": 0.25,
        "allowed_activations": ["relu", "leaky_relu"],
    }
    cfg.update(overrides)
    return cfg


def make_gan_config(**overrides: Any) -> Dict[str, Any]:
    """
    Minimal valid GAN component config dict.

    Args:
        **overrides: Key-value pairs to override defaults.
    """
    cfg: Dict[str, Any] = {
        "num_classes": 4,
        "in_channels": 1,
        "base_filters": 4,
        "depth": 2,
        "lambda_dice": 1.0,
        "lambda_gan": 0.1,
        "lambda_laplace": 0.05,
        "gan_mode": "lsgan",
        "learning_rate_g": 1e-4,
        "learning_rate_d": 1e-4,
        "entropy_threshold": 0.5,
        "collapse_window": 3,
    }
    cfg.update(overrides)
    return cfg


def make_laplace_config(**overrides: Any) -> Dict[str, Any]:
    """
    Minimal valid Laplace component config dict.

    Args:
        **overrides: Key-value pairs to override defaults.
    """
    cfg: Dict[str, Any] = {
        "num_iterations": 5,
        "lambda_smooth": 0.5,
        "algorithm": "taubin",
        "target_genus": 0,
        "max_holes": 0,
        "voxel_size_mm": 1.0,
        "iso_threshold": 0.5,
    }
    cfg.update(overrides)
    return cfg
