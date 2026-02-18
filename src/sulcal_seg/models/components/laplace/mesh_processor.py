"""Mesh extraction and rasterisation utilities (stubs)."""
from typing import Any, Dict, Tuple

import numpy as np


class MeshProcessor:
    """
    Extracts a surface mesh from a segmentation probability map and
    rasterises meshes back to voxel grids.

    Requires: scikit-image (marching cubes) + trimesh.
    Both operations are stubs pending data/GPU availability.
    """

    def __init__(self, iso_threshold: float = 0.5, voxel_size_mm: float = 1.0) -> None:
        """
        Args:
            iso_threshold: Iso-surface level for marching cubes.
            voxel_size_mm: Voxel spacing in mm (isotropic assumed).
        """
        self.iso_threshold = iso_threshold
        self.voxel_size_mm = voxel_size_mm

    def extract_mesh(
        self,
        probability_map: np.ndarray,
        label_index: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a triangle mesh from a class probability map using marching cubes.

        Args:
            probability_map: (C, D, H, W) probability array (0–1).
            label_index: Class index to extract the surface for.

        Returns:
            (vertices, faces) — (N, 3) and (M, 3) numpy arrays.

        Raises:
            NotImplementedError: Always (stub).
        """
        # TODO: Extract probability_map[label_index], apply marching cubes
        #   via skimage.measure.marching_cubes(volume, level=self.iso_threshold),
        #   scale vertices by voxel_size_mm, return (vertices, faces).
        raise NotImplementedError(
            "MeshProcessor.extract_mesh() requires skimage.measure.marching_cubes. "
            "Implement mesh extraction here."
        )

    def rasterize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Rasterise a triangle mesh back to a binary voxel volume.

        Args:
            vertices: (N, 3) mesh vertices in voxel coordinates.
            faces: (M, 3) triangle face indices.
            output_shape: (D, H, W) target voxel grid shape.

        Returns:
            Binary volume (D, H, W) with 1 inside the mesh, 0 outside.

        Raises:
            NotImplementedError: Always (stub).
        """
        # TODO: Use trimesh.Trimesh(vertices, faces).voxelized(pitch=voxel_size_mm)
        #   or a ray-casting approach to fill the interior.
        raise NotImplementedError(
            "MeshProcessor.rasterize_mesh() requires trimesh. "
            "Implement mesh rasterisation here."
        )
