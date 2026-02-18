"""Data quality validation for MRI volumes."""
from typing import Dict, List

import numpy as np


class DataValidator:
    """
    Validates MRI volumes and segmentation labels before training.
    """

    def validate_image(self, image: np.ndarray) -> List[str]:
        """
        Validate an MRI image array.

        Returns:
            List of error messages. Empty list means valid.
        """
        errors = []
        if image.ndim != 3:
            errors.append(f"Expected 3D image, got {image.ndim}D with shape {image.shape}")
        if np.isnan(image).any():
            errors.append("Image contains NaN values")
        if np.isinf(image).any():
            errors.append("Image contains Inf values")
        if image.std() < 1e-8:
            errors.append("Image has near-zero variance (possibly all-zero or uniform)")
        return errors

    def validate_label(self, label: np.ndarray, num_labels: int) -> List[str]:
        """
        Validate a segmentation label array.

        Args:
            label: 3D integer label array
            num_labels: Expected number of distinct labels (including background=0)

        Returns:
            List of error messages. Empty list means valid.
        """
        errors = []
        if label.ndim != 3:
            errors.append(f"Expected 3D label, got {label.ndim}D")
        if np.isnan(label).any():
            errors.append("Label contains NaN values")
        min_val = int(label.min())
        max_val = int(label.max())
        if min_val < 0:
            errors.append(f"Label contains negative values (min={min_val})")
        if max_val >= num_labels:
            errors.append(f"Label contains value {max_val} >= num_labels {num_labels}")
        return errors

    def validate_pair(
        self, image: np.ndarray, label: np.ndarray
    ) -> List[str]:
        """Validate that image and label have matching shapes."""
        errors = []
        if image.shape != label.shape:
            errors.append(
                f"Image shape {image.shape} does not match label shape {label.shape}"
            )
        return errors

    def check_all(
        self,
        image: np.ndarray,
        label: np.ndarray,
        num_labels: int = 41,
    ) -> Dict[str, List[str]]:
        """
        Run all validation checks.

        Returns:
            Dict mapping check name to list of errors (empty = passed)
        """
        return {
            "image": self.validate_image(image),
            "label": self.validate_label(label, num_labels),
            "pair": self.validate_pair(image, label),
        }
