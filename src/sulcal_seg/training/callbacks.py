"""Training callbacks: early stopping, checkpointing."""
from pathlib import Path
from typing import Any, Dict, Optional


class EarlyStopping:
    """
    Monitor a validation metric and stop training when it stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum absolute change to count as improvement.
        mode: 'min' (lower is better) or 'max' (higher is better).
        verbose: Log stopping decision to stdout.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = False,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self._counter: int = 0
        self._best: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """
        Update state with new metric value.

        Args:
            metric: Current validation metric.

        Returns:
            True if training should stop, False otherwise.
        """
        if self._best is None:
            self._best = metric
            return False

        if self.mode == "max":
            improved = metric > self._best + self.min_delta
        else:
            improved = metric < self._best - self.min_delta

        if improved:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: no improvement for {self._counter}/{self.patience} epochs "
                    f"(best={self._best:.4f}, current={metric:.4f})"
                )

        if self._counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"EarlyStopping: triggered after {self._counter} stagnant epochs.")

        return self.should_stop

    def reset(self) -> None:
        """Reset counter (e.g., after a learning rate restart)."""
        self._counter = 0
        self.should_stop = False


class CheckpointCallback:
    """
    Save model checkpoints during training.

    Saves:
        - Best checkpoint (lowest/highest metric seen so far)
        - Periodic checkpoints every `save_every` epochs

    Args:
        checkpoint_dir: Directory to write checkpoint files.
        save_every: Save a checkpoint every N epochs (0 = disable periodic saves).
        mode: 'min' or 'max' for best-metric tracking.
        verbose: Log save events.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 10,
        mode: str = "max",
        verbose: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.mode = mode
        self.verbose = verbose

        self._best_metric: Optional[float] = None
        self._epoch: int = 0
        self.best_checkpoint_path: Optional[Path] = None

    def __call__(
        self,
        epoch: int,
        metric: float,
        state: Dict[str, Any],
        save_fn: Any,
    ) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch index (0-based).
            metric: Current validation metric.
            state: Serialisable state dict to save.
            save_fn: Callable(state, path) that writes the checkpoint to disk.
        """
        self._epoch = epoch

        # Best checkpoint
        is_best = self._is_best(metric)
        if is_best:
            self._best_metric = metric
            path = self.checkpoint_dir / "best.pt"
            save_fn(state, path)
            self.best_checkpoint_path = path
            if self.verbose:
                print(f"CheckpointCallback: saved best checkpoint (metric={metric:.4f})")

        # Periodic checkpoint
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_fn(state, path)
            if self.verbose:
                print(f"CheckpointCallback: saved periodic checkpoint at epoch {epoch}")

    def _is_best(self, metric: float) -> bool:
        if self._best_metric is None:
            return True
        if self.mode == "max":
            return metric > self._best_metric
        return metric < self._best_metric
