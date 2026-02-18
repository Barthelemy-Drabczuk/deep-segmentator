"""Topology validation for segmentation meshes."""
from typing import Any, Dict, List, Tuple

import numpy as np


class TopologyValidator:
    """
    Validates topological properties of a sulcal surface mesh.

    Checks implemented:
        - Vertex valence histogram (fully computable from faces alone)
        - Genus-0 check via Euler characteristic (V - E + F = 2 for genus-0)

    Checks that require trimesh (stubs):
        - Hole detection
        - Non-manifold edge detection
    """

    def __init__(self, target_genus: int = 0, max_holes: int = 0) -> None:
        """
        Args:
            target_genus: Expected topological genus (0 = sphere-like).
            max_holes: Maximum allowed number of boundary loops.
        """
        self.target_genus = target_genus
        self.max_holes = max_holes

    def compute_valence_histogram(
        self, faces: np.ndarray, num_vertices: int
    ) -> np.ndarray:
        """
        Compute vertex valence (number of adjacent faces) for each vertex.

        Args:
            faces: (M, 3) face index array.
            num_vertices: Total number of vertices V.

        Returns:
            (V,) integer array of valence counts.
        """
        valence = np.zeros(num_vertices, dtype=np.int32)
        for face in faces:
            for vertex_idx in face:
                valence[vertex_idx] += 1
        return valence

    def check_euler_characteristic(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute the Euler characteristic χ = V - E + F and infer genus.

        For a closed orientable surface: χ = 2 - 2g (g = genus).

        Args:
            vertices: (N, 3) vertex positions.
            faces: (M, 3) face indices.

        Returns:
            Dict with keys:
                - 'V': number of vertices
                - 'E': number of edges
                - 'F': number of faces
                - 'euler': χ = V - E + F
                - 'genus': inferred genus (g = (2 - χ) / 2)
                - 'is_target_genus': bool
        """
        V = len(vertices)
        F = len(faces)

        # Count unique edges (each edge shared by ≤2 faces)
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges.add(edge)
        E = len(edges)

        euler = V - E + F
        genus = (2 - euler) // 2

        return {
            "V": V,
            "E": E,
            "F": F,
            "euler": euler,
            "genus": int(genus),
            "is_target_genus": int(genus) == self.target_genus,
        }

    def check_holes(self, vertices: np.ndarray, faces: np.ndarray) -> int:
        """
        Count boundary loops (open holes) in the mesh.

        Args:
            vertices: (N, 3) vertex positions.
            faces: (M, 3) face indices.

        Returns:
            Number of boundary loops.

        Raises:
            NotImplementedError: Full implementation requires trimesh.
        """
        # TODO: Use trimesh.Trimesh(vertices, faces).is_watertight or
        #   detect boundary edges (edges referenced by exactly one face).
        raise NotImplementedError(
            "check_holes() requires trimesh for robust boundary loop detection."
        )

    def validate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run all available topology checks.

        Returns:
            Dict with Euler characteristic results and a 'passed' boolean.
            Hole check is skipped (returns None) since it requires trimesh.
        """
        euler_result = self.check_euler_characteristic(vertices, faces)
        valence = self.compute_valence_histogram(faces, len(vertices))

        return {
            "euler": euler_result,
            "valence_mean": float(valence.mean()),
            "valence_max": int(valence.max()),
            "holes": None,  # Requires trimesh — see check_holes()
            "passed": euler_result["is_target_genus"],
        }
