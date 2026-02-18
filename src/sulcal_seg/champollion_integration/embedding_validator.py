"""Validate Champollion embeddings derived from sulcal segmentations (stub)."""
from typing import Any, Dict

import numpy as np


class EmbeddingValidator:
    """
    Validates that sulcal embeddings preserve expected properties:
        - Age correlation in lifespan embeddings
        - Sex/scanner harmonisation
        - Cross-cohort consistency

    This class is a stub — requires Champollion embeddings and metadata.
    """

    def validate_age_correlation(
        self,
        embeddings: np.ndarray,
        ages: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute correlation between embedding principal components and age.

        Args:
            embeddings: (N, D) embedding matrix.
            ages: (N,) age values.

        Returns:
            Dict with 'pearson_r', 'spearman_r', 'p_value'.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "EmbeddingValidator.validate_age_correlation() requires champollion_utils."
        )

    def validate_cross_cohort(
        self,
        embeddings_by_cohort: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Check that embeddings from different cohorts occupy similar manifold regions.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "EmbeddingValidator.validate_cross_cohort() requires champollion_utils."
        )
