"""Visualization utilities for training curves and segmentation results."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_training_curves(
    losses: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    title: str = "Training Curves",
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        losses: Dict mapping metric name to list of values per epoch
        output_path: If provided, save plot to this path
        title: Plot title
    """
    # TODO: Implement training curve visualization
    try:
        import matplotlib.pyplot as plt

        n_plots = len(losses)
        fig, axes = plt.subplots(1, max(1, n_plots), figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, losses.items()):
            ax.plot(values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True)

        fig.suptitle(title)
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    except ImportError:
        pass


def plot_segmentation_overlay(
    image: np.ndarray,
    label: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot 2D axial/coronal/sagittal slice with segmentation overlay.

    Args:
        image: 3D MRI image [H, W, D]
        label: 3D ground truth label [H, W, D]
        prediction: 3D predicted label [H, W, D] (optional)
        slice_idx: Slice index to display (default: middle slice)
        axis: Axis along which to take slice (0=sagittal, 1=coronal, 2=axial)
        output_path: If provided, save to this path
    """
    # TODO: Implement 2D slice overlay visualization
    try:
        import matplotlib.pyplot as plt

        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        def get_slice(vol: np.ndarray) -> np.ndarray:
            return np.take(vol, slice_idx, axis=axis)

        n_cols = 3 if prediction is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(get_slice(image), cmap="gray")
        axes[0].set_title("MRI Image")
        axes[0].axis("off")

        axes[1].imshow(get_slice(label), cmap="nipy_spectral")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        if prediction is not None:
            axes[2].imshow(get_slice(prediction), cmap="nipy_spectral")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    except ImportError:
        pass


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot ablation study results as a grouped bar chart.

    Args:
        results: Dict mapping ablation name -> dict of metric -> value
        metrics: List of metric names to plot
        output_path: If provided, save to this path
    """
    # TODO: Implement ablation results bar chart
    pass
