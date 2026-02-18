"""Main trainer for sulcal segmentation models (stub)."""
from typing import Any, Callable, Dict, List, Optional

from sulcal_seg.models.full_model import SulcalSegmentationModel
from sulcal_seg.training.callbacks import CheckpointCallback, EarlyStopping


class Trainer:
    """
    Orchestrates model training with configurable callbacks.

    The core training loop (``train``, ``_run_epoch``) is a stub —
    it requires GPU and real data loaders. The callback integration
    and Optuna hook are fully implemented.

    Args:
        model: The assembled :class:`SulcalSegmentationModel`.
        config: Training configuration dict.
        callbacks: List of callback instances (EarlyStopping, CheckpointCallback, …).
    """

    def __init__(
        self,
        model: SulcalSegmentationModel,
        config: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        self._optuna_callback: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_optuna_callback(self, trial: Any) -> None:
        """
        Register an Optuna trial for mid-training pruning.

        The callback is invoked at the end of each epoch with the current
        validation metric. If Optuna decides to prune, a
        ``optuna.exceptions.TrialPruned`` exception is raised.

        Args:
            trial: An ``optuna.Trial`` object.
        """
        import optuna  # noqa: PLC0415 — deferred import

        def _callback(epoch: int, metric: float) -> None:
            trial.report(metric, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned(
                    f"Trial pruned at epoch {epoch} (metric={metric:.4f})"
                )

        self._optuna_callback = _callback

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Override epoch count from config.

        Returns:
            Dict of final validation metrics.

        Raises:
            NotImplementedError: Always (stub — requires GPU + data).
        """
        # TODO: Implement the training loop:
        #   for epoch in range(num_epochs):
        #       train_metrics = self._run_epoch(train_loader, train=True)
        #       val_metrics = self._run_epoch(val_loader, train=False)
        #       for cb in self.callbacks:
        #           cb(epoch, val_metrics['val_dice'], state, save_fn)
        #       if self._optuna_callback:
        #           self._optuna_callback(epoch, val_metrics['val_dice'])
        #       if any EarlyStopping.should_stop: break
        raise NotImplementedError(
            "Trainer.train() requires GPU and real data loaders."
        )

    def _run_epoch(
        self,
        loader: Any,
        train: bool = True,
    ) -> Dict[str, float]:
        """
        Run one epoch (forward + optional backward pass).

        Args:
            loader: DataLoader for this epoch.
            train: If True, perform backward pass and optimiser step.

        Returns:
            Dict of metrics for this epoch.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "Trainer._run_epoch() requires GPU and real data loaders."
        )
