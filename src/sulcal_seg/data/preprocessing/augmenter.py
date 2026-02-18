"""Data augmentation for 3D MRI volumes."""
from typing import Tuple

import numpy as np


class VolumeAugmenter:
    """
    Data augmentation for 3D MRI volumes and their corresponding labels.

    Applies stochastic transforms that preserve anatomical validity:
    - Random horizontal flip (left-right)
    - Random intensity scaling (gamma augmentation)
    - TODO: Rotation (requires scipy.ndimage.rotate)
    - TODO: Elastic deformation (requires scipy.ndimage)
    - TODO: Scale/zoom (requires scipy.ndimage.zoom)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_range_deg: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        gamma_range: Tuple[float, float] = (0.85, 1.15),
        seed: int = 42,
    ):
        self.flip_prob = flip_prob
        self.rotation_range_deg = rotation_range_deg
        self.scale_range = scale_range
        self.gamma_range = gamma_range
        self._rng = np.random.default_rng(seed)

    def augment(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentation to an image-label pair.

        Args:
            image: 3D T1 MRI array [H, W, D]
            label: 3D integer label array [H, W, D]

        Returns:
            (augmented_image, augmented_label) as np.ndarray pair
        """
        image = image.astype(np.float32)
        label = label.copy()

        # Random horizontal flip (axis 0 = left-right)
        if self._rng.random() < self.flip_prob:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        # Random gamma intensity augmentation (image only, not label)
        gamma = self._rng.uniform(*self.gamma_range)
        image_min = image.min()
        image_range = image.max() - image_min
        if image_range > 1e-8:
            normalized = (image - image_min) / image_range
            image = np.power(np.clip(normalized, 0, 1), gamma) * image_range + image_min

        # TODO: Random rotation (requires scipy)
        # from scipy.ndimage import rotate as ndrotate
        # angle = self._rng.uniform(-self.rotation_range_deg, self.rotation_range_deg)
        # image = ndrotate(image, angle, axes=(0, 1), reshape=False)
        # label = ndrotate(label, angle, axes=(0, 1), reshape=False, order=0)

        # TODO: Elastic deformation (requires scipy.ndimage)

        return image, label

    def set_seed(self, seed: int) -> None:
        """Reset the random state for reproducibility."""
        self._rng = np.random.default_rng(seed)
