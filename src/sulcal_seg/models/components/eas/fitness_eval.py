"""Fitness evaluation for architecture genomes (proxy training stub)."""
from typing import Any, Dict, Optional

from sulcal_seg.models.components.eas.search_space import ArchitectureGenome


class FitnessEvaluator:
    """
    Evaluates an ArchitectureGenome by training a proxy U-Net for a few
    epochs on the training loader and returning its validation Dice score.

    This class is a stub — the real implementation requires GPU and data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: EASComponentConfig dict (num_classes, proxy_epochs, etc.)
        """
        self.config = config

    def evaluate(
        self,
        genome: ArchitectureGenome,
        train_loader: Any,
        val_loader: Any,
        device: Optional[str] = None,
    ) -> float:
        """
        Train a proxy network derived from `genome` and return its validation Dice.

        Args:
            genome: Candidate architecture to evaluate.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Torch device string (e.g. "cuda:0"). Defaults to CPU if None.

        Returns:
            Validation macro-Dice score in [0, 1].

        Raises:
            NotImplementedError: Always (stub — requires real data + GPU).
        """
        # TODO: Build a lightweight U-Net from genome, train for proxy_epochs,
        # compute val Dice and return it as fitness.
        raise NotImplementedError(
            "FitnessEvaluator.evaluate() requires GPU and real data loaders. "
            "Implement proxy training loop here."
        )
