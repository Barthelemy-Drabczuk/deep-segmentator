"""Checkpoint saving and loading utilities."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .logging import get_logger

_logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic rotation.

    Saves the N most recent checkpoints and always keeps the best model.
    """

    def __init__(self, checkpoint_dir: Path, max_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._saved: List[Path] = []

    def save(self, state: Dict[str, Any], filename: str) -> Path:
        """
        Save a checkpoint state dict to file.

        Args:
            state: Dictionary to save (should include 'epoch', 'metrics', and optionally 'model_state_dict')
            filename: Filename (e.g., 'checkpoint_epoch_10.pt' or 'best_model.pt')

        Returns:
            Path to saved checkpoint
        """
        path = self.checkpoint_dir / filename
        torch.save(state, path)

        # Track regular checkpoints (not the 'best' checkpoint)
        if "best" not in filename:
            self._saved.append(path)
            if len(self._saved) > self.max_keep:
                oldest = self._saved.pop(0)
                if oldest.exists():
                    oldest.unlink()
                    _logger.debug(f"Removed old checkpoint: {oldest}")

        _logger.info(f"Checkpoint saved: {path}")
        return path

    def load(self, path: Path, map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint from file.

        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda')

        Returns:
            Loaded state dictionary
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location)
        _logger.info(f"Checkpoint loaded: {path}")
        return state

    def get_latest(self) -> Optional[Path]:
        """Return path to the most recent checkpoint, or None if none exist."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def get_best(self) -> Optional[Path]:
        """Return path to the best model checkpoint, or None if none exist."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None
