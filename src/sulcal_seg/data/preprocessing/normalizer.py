"""Intensity normalization for MRI volumes."""
from typing import Optional

import numpy as np


class IntensityNormalizer:
    """
    Intensity normalization for 3D MRI volumes.

    Supports z-score normalization (default), min-max scaling,
    and percentile-based normalization.
    """

    VALID_METHODS = ("zscore", "minmax", "percentile")

    def __init__(self, method: str = "zscore"):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {method!r}")
        self.method = method

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize a 3D MRI volume.

        Args:
            image: 3D array [H, W, D]
            mask: Optional binary mask [H, W, D] — stats computed within mask only

        Returns:
            Normalized array of same shape and dtype float32
        """
        image = image.astype(np.float32)
        if self.method == "zscore":
            return self._zscore(image, mask)
        elif self.method == "minmax":
            return self._minmax(image)
        else:
            return self._percentile(image)

    def _zscore(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Z-score normalization: (I - mean) / std"""
        if mask is not None:
            values = image[mask > 0]
        else:
            values = image.ravel()

        mean = float(values.mean())
        std = float(values.std())

        if std < 1e-8:
            return image - mean
        return (image - mean) / std

    def _minmax(self, image: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        mn = float(image.min())
        mx = float(image.max())
        if mx - mn < 1e-8:
            return np.zeros_like(image)
        return (image - mn) / (mx - mn)

    def _percentile(
        self, image: np.ndarray, low: float = 0.5, high: float = 99.5
    ) -> np.ndarray:
        """Percentile clipping followed by min-max normalization."""
        p_low, p_high = float(np.percentile(image, low)), float(np.percentile(image, high))
        clipped = np.clip(image, p_low, p_high)
        return self._minmax(clipped)
