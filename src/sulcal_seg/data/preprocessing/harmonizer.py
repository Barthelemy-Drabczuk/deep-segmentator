"""
Multi-cohort harmonization.

TODO: Implement ComBat harmonization using neuroHarmonize.
    pip install neuroHarmonize  (not available on conda-forge, use pypi)

Reference:
    Fortin et al. (2018). Harmonization of cortical thickness measurements
    across scanners and sites. NeuroImage.
"""
from typing import Optional

import numpy as np


class CohortHarmonizer:
    """
    ComBat-style harmonization to remove scanner/site batch effects.

    Fitted on training data, applied to all cohorts during inference.
    """

    def __init__(self, method: str = "combat"):
        self.method = method
        self._is_fitted = False

    def fit(self, data: np.ndarray, covariates: np.ndarray) -> "CohortHarmonizer":
        """
        Fit the harmonization model.

        Args:
            data: Feature matrix [n_subjects, n_features]
            covariates: Covariate matrix [n_subjects, n_covariates]
                        (should include batch/site column)

        TODO: Implement using neuroHarmonize.harmonizationLearn
        """
        raise NotImplementedError(
            "TODO: Implement ComBat harmonization. "
            "Install neuroHarmonize: pip install neuroHarmonize"
        )

    def transform(self, data: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """
        Apply harmonization to new data.

        Args:
            data: Feature matrix [n_subjects, n_features]
            covariates: Covariate matrix [n_subjects, n_covariates]

        TODO: Implement using neuroHarmonize.harmonizationApply
        """
        raise NotImplementedError(
            "TODO: Implement ComBat harmonization transform. "
            "See COMPLETE_MODEL_SPECIFICATION.md §Data Harmonization."
        )
