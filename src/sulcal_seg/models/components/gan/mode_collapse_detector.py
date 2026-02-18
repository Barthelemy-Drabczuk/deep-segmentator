"""Mode collapse detection for the GAN segmentation component."""
from typing import Any, Dict, Optional

import torch


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean Shannon entropy of a batch of probability maps.

    H = -sum_c p_c * log(p_c + ε)

    averaged over all spatial locations and batch elements.

    Args:
        probs: Softmax probabilities (B, C, D, H, W).

    Returns:
        Scalar mean entropy tensor.
    """
    eps = 1e-8
    # H per voxel: -sum_c p_c * log(p_c)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)  # (B, D, H, W)
    return entropy.mean()


def compute_fid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> float:
    """
    Compute the Fréchet Inception Distance (FID) between real and fake feature sets.

    Args:
        real_features: Feature matrix from real samples (N, D).
        fake_features: Feature matrix from generated samples (M, D).

    Returns:
        FID score (lower = more similar distributions).

    Raises:
        NotImplementedError: Always (stub — requires scipy.linalg.sqrtm and
            a pre-trained feature extractor for 3D volumes).
    """
    # TODO: Implement FID for 3D medical volumes:
    #   1. Compute mean and covariance for real_features and fake_features
    #   2. FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r @ Σ_f)^0.5)
    raise NotImplementedError(
        "compute_fid() is not yet implemented. "
        "Requires a 3D feature extractor (e.g., pre-trained encoder) and "
        "scipy.linalg.sqrtm for matrix square root."
    )


class ModeCollapseDetector:
    """
    Monitors generator output entropy over training to detect mode collapse.

    Mode collapse is flagged when the mean output entropy falls below
    `entropy_threshold` for `window` consecutive evaluation steps.

    Args:
        entropy_threshold: Entropy below this value triggers a warning.
        window: Number of consecutive low-entropy steps to confirm collapse.
    """

    def __init__(self, entropy_threshold: float = 0.5, window: int = 10) -> None:
        self.entropy_threshold = entropy_threshold
        self.window = window
        self._history: list = []
        self._step: int = 0

    def update(self, probs: torch.Tensor) -> Dict[str, Any]:
        """
        Record current entropy and check for collapse.

        Args:
            probs: Generator output probabilities (B, C, D, H, W).

        Returns:
            Dict with keys:
                - 'entropy': Current mean entropy value (float)
                - 'collapsed': True if collapse detected (bool)
                - 'step': Current monitoring step (int)
        """
        with torch.no_grad():
            entropy_val = compute_entropy(probs).item()

        self._history.append(entropy_val)
        self._step += 1

        # Keep only the last `window` values
        if len(self._history) > self.window:
            self._history.pop(0)

        collapsed = (
            len(self._history) == self.window
            and all(e < self.entropy_threshold for e in self._history)
        )

        return {
            "entropy": entropy_val,
            "collapsed": collapsed,
            "step": self._step,
        }

    def reset(self) -> None:
        """Reset history (e.g., at the start of a new epoch)."""
        self._history.clear()
        self._step = 0
