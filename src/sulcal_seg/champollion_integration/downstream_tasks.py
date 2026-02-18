"""Downstream tasks using Champollion sulcal embeddings (stub)."""
from typing import Any, Dict

import numpy as np


class DownstreamTaskRunner:
    """
    Runs downstream prediction tasks on Champollion sulcal embeddings:
        - Age prediction (regression)
        - Neurological disorder classification (ASD, dementia)
        - Lifespan trajectory modelling

    This class is a stub — requires pre-computed embeddings.
    """

    def predict_age(
        self,
        embeddings: np.ndarray,
        ages: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train and evaluate a brain-age prediction model.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "DownstreamTaskRunner.predict_age() requires champollion_utils."
        )

    def classify_disorder(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train and evaluate a disorder classification model.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "DownstreamTaskRunner.classify_disorder() requires champollion_utils."
        )
