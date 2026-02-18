"""Laplace geometry constraint component engine."""
from typing import Any, Dict

import numpy as np

from sulcal_seg.models.components.base_component import BaseComponent
from sulcal_seg.models.components.laplace.laplace_smoother import LaplaceSmoother
from sulcal_seg.models.components.laplace.mesh_processor import MeshProcessor
from sulcal_seg.models.components.laplace.validators import TopologyValidator


_REQUIRED_KEYS = frozenset({"num_iterations", "lambda_smooth"})


class LaplaceEngine(BaseComponent):
    """
    Laplace geometry constraint component.

    Post-processes GAN segmentation outputs by:
    1. Extracting a surface mesh via marching cubes
    2. Applying Taubin Laplacian smoothing
    3. Rasterising the smoothed mesh back to voxel space
    4. Validating mesh topology

    The full pipeline is stubbed — it requires trimesh and scikit-image.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.smoother = LaplaceSmoother(
            num_iterations=config.get("num_iterations", 30),
            lambda_smooth=config.get("lambda_smooth", 0.5),
            algorithm=config.get("algorithm", "taubin"),
        )
        self.mesh_processor = MeshProcessor(
            iso_threshold=config.get("iso_threshold", 0.5),
            voxel_size_mm=config.get("voxel_size_mm", 1.0),
        )
        self.topology_validator = TopologyValidator(
            target_genus=config.get("target_genus", 0),
            max_holes=config.get("max_holes", 0),
        )

    # ------------------------------------------------------------------
    # BaseComponent interface
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = _REQUIRED_KEYS - set(self.config.keys())
        if missing:
            raise ValueError(f"LaplaceEngine config missing keys: {missing}")
        if not (0.0 < self.config["lambda_smooth"] <= 1.0):
            raise ValueError("lambda_smooth must be in (0, 1]")
        if self.config["num_iterations"] < 1:
            raise ValueError("num_iterations must be >= 1")

    def train(self, train_loader: Any, val_loader: Any) -> Dict[str, float]:
        """
        The Laplace component has no trainable parameters — it is a
        differentiable geometry post-processor.

        Returns an empty metrics dict.
        """
        # The Laplace component does not have its own training loop.
        # It is applied to the output of the GAN component.
        return {"val_topology_score": 0.0}

    def forward(self, probability_map: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Apply Laplace-Beltrami post-processing to a segmentation probability map.

        Args:
            probability_map: (C, D, H, W) numpy array of class probabilities.

        Raises:
            NotImplementedError: Always (stub — requires trimesh + skimage).
        """
        # TODO: For each class label:
        #   1. vertices, faces = self.mesh_processor.extract_mesh(probability_map, c)
        #   2. smooth_v, _ = self.smoother.smooth(vertices, faces)
        #   3. mask = self.mesh_processor.rasterize_mesh(smooth_v, faces, output_shape)
        #   Combine per-class masks into a final label map.
        raise NotImplementedError(
            "LaplaceEngine.forward() requires trimesh and skimage. "
            "Implement mesh-based post-processing here."
        )
