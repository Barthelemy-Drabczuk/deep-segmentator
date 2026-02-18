"""Optuna pruner wrappers for early stopping of unpromising trials."""
from typing import Any, Optional


def create_hyperband_pruner(
    min_resource: int = 5,
    max_resource: int = 100,
    reduction_factor: int = 3,
) -> Any:
    """
    Create an Optuna HyperbandPruner.

    Args:
        min_resource: Minimum number of training epochs before pruning.
        max_resource: Maximum number of training epochs.
        reduction_factor: Bracket reduction factor (η in Hyperband).

    Returns:
        optuna.pruners.HyperbandPruner instance.
    """
    import optuna  # noqa: PLC0415

    return optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )


def create_median_pruner(
    n_startup_trials: int = 5,
    n_warmup_steps: int = 10,
) -> Any:
    """
    Create an Optuna MedianPruner.

    Args:
        n_startup_trials: Trials before pruning starts.
        n_warmup_steps: Epochs before pruning starts per trial.

    Returns:
        optuna.pruners.MedianPruner instance.
    """
    import optuna  # noqa: PLC0415

    return optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )
