"""Convert sulcal segmentation outputs to Champollion embedding space (stub)."""
from typing import Any, Dict, Optional

import numpy as np


class SegmentationToEmbeddings:
    """
    Projects sulcal segmentation label maps into Champollion's embedding space
    for downstream lifespan trajectory modelling.

    This class is a stub — it requires a pre-trained Champollion encoder
    and access to the champollion_utils package.
    """

    def __init__(self, champollion_model: Any, config: Dict[str, Any]) -> None:
        """
        Args:
            champollion_model: Pre-trained Champollion model instance.
            config: Configuration dict with embedding parameters.
        """
        self.champollion_model = champollion_model
        self.config = config

    def encode(
        self,
        segmentation: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Encode a segmentation map to a Champollion embedding vector.

        Args:
            segmentation: Integer label map (D, H, W).
            metadata: Optional subject metadata (age, sex, scanner, etc.).

        Returns:
            Embedding vector (embedding_dim,).

        Raises:
            NotImplementedError: Always (stub).
        """
        # TODO: Extract sulcal morphology features from segmentation,
        #   pass to champollion_model.encode(), return embedding.
        raise NotImplementedError(
            "SegmentationToEmbeddings.encode() requires champollion_utils."
        )

    def encode_batch(
        self,
        segmentations: np.ndarray,
        metadata_list: Optional[list] = None,
    ) -> np.ndarray:
        """
        Encode a batch of segmentations.

        Args:
            segmentations: (N, D, H, W) integer label maps.
            metadata_list: Optional list of N metadata dicts.

        Returns:
            (N, embedding_dim) embedding matrix.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "SegmentationToEmbeddings.encode_batch() requires champollion_utils."
        )
