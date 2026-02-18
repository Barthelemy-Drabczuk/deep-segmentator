"""Abstract base class for all model components (EAS, GAN, Laplace)."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch


class BaseComponent(ABC):
    """
    Abstract base for all sulcal segmentation model components.

    Each component (EAS, GAN, Laplace) inherits from this class
    and can be trained independently, then assembled via ModelBuilder.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration keys and values. Raise ValueError if invalid."""

    @abstractmethod
    def train(self, train_loader, val_loader) -> Dict[str, float]:
        """
        Train this component. Returns dict of validation metrics.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Dict mapping metric name to value (e.g., {'val_dice': 0.95})
        """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through this component."""

    def save(self, checkpoint_path) -> None:
        """
        Save component state to file.

        Args:
            checkpoint_path: Path to save checkpoint (.pt file)
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: Dict[str, Any] = {"config": self.config}
        # Save module weights if this component has nn.Module attributes
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                state[f"{attr_name}_state_dict"] = attr.state_dict()
        torch.save(state, path)

    def load(self, checkpoint_path) -> None:
        """
        Load component state from file.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        path = Path(checkpoint_path)
        state = torch.load(path, map_location="cpu")
        for key, value in state.items():
            if key.endswith("_state_dict"):
                attr_name = key.replace("_state_dict", "")
                attr = getattr(self, attr_name, None)
                if isinstance(attr, torch.nn.Module):
                    attr.load_state_dict(value)
