"""GAN segmentation component engine."""
from typing import Any, Dict

import torch

from sulcal_seg.models.components.base_component import BaseComponent
from sulcal_seg.models.components.gan.discriminator import Discriminator
from sulcal_seg.models.components.gan.dual_label_trainer import DualLabelTrainer
from sulcal_seg.models.components.gan.generator import Generator
from sulcal_seg.models.components.gan.mode_collapse_detector import ModeCollapseDetector


_REQUIRED_KEYS = frozenset({"num_classes", "in_channels"})


class GANEngine(BaseComponent):
    """
    Multi-label GAN segmentation component.

    Contains a Generator (U-Net) and Discriminator trained adversarially
    against both Morphologist and FreeSurfer label sets to prevent mode collapse.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.generator = Generator(
            in_channels=config.get("in_channels", 1),
            num_classes=config["num_classes"],
            base_filters=config.get("base_filters", 32),
            depth=config.get("depth", 4),
            activation="relu",
        )
        # Discriminator input = image channels + num_classes (softmax concat)
        disc_in = config.get("in_channels", 1) + config["num_classes"]
        self.discriminator = Discriminator(
            in_channels=disc_in,
            base_filters=config.get("base_filters", 64),
        )
        self.collapse_detector = ModeCollapseDetector(
            entropy_threshold=config.get("entropy_threshold", 0.5),
            window=config.get("collapse_window", 10),
        )
        self.trainer = DualLabelTrainer(self.generator, self.discriminator, config)

    # ------------------------------------------------------------------
    # BaseComponent interface
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = _REQUIRED_KEYS - set(self.config.keys())
        if missing:
            raise ValueError(f"GANEngine config missing keys: {missing}")
        if self.config["num_classes"] < 2:
            raise ValueError("num_classes must be >= 2")
        if self.config["in_channels"] < 1:
            raise ValueError("in_channels must be >= 1")

    def train(self, train_loader: Any, val_loader: Any) -> Dict[str, float]:
        """
        Train the GAN for one full pass over train_loader.

        Raises:
            NotImplementedError: Always (stub — requires GPU + data loaders).
        """
        # TODO: Iterate over train_loader, call self.trainer.train_step(),
        #       monitor self.collapse_detector.update() after each step,
        #       evaluate on val_loader, return metrics dict.
        raise NotImplementedError(
            "GANEngine.train() requires GPU and real data loaders."
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Run generator forward pass (inference mode).

        Args:
            x: Input MRI batch (B, 1, D, H, W).

        Returns:
            Segmentation logits (B, num_classes, D, H, W).
        """
        return self.generator(x)
