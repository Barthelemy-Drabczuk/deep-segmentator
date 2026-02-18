"""HOSVD-based architecture breeding operator for EAS.

Tucker decomposition (Higher-Order SVD) is used to extract low-rank
representations of weight tensors from two parent architectures, then
reconstruct a child with blended structure.

Requires: tensorly (optional — guarded by HAS_TENSORLY flag).
If tensorly is not installed, breed_offspring() raises ImportError.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from sulcal_seg.models.components.eas.search_space import ArchitectureGenome

try:
    import tensorly as tl
    from tensorly.decomposition import tucker

    HAS_TENSORLY = True
except ImportError:  # pragma: no cover
    HAS_TENSORLY = False

if TYPE_CHECKING:  # pragma: no cover
    pass


class HOSVDBreeder:
    """
    Breeds two parent ArchitectureGenomes using HOSVD-guided weight tensor blending.

    The breeding process:
    1. Convert filter count lists of both parents into weight tensors
    2. Apply Tucker decomposition to extract core + factor matrices
    3. Reconstruct a child tensor by interpolating core tensors at `alpha`
    4. Decode the child tensor back to a genome
    """

    def __init__(self, rank_ratios: Tuple[float, float, float] = (0.7, 0.7, 0.7)) -> None:
        """
        Args:
            rank_ratios: Fraction of modes to keep in Tucker decomposition.
                         Three values for up to 3 tensor modes.
        """
        self.rank_ratios = rank_ratios

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def breed_offspring(
        self,
        parent_a: ArchitectureGenome,
        parent_b: ArchitectureGenome,
        alpha: float = 0.5,
        genome_id: Optional[int] = None,
    ) -> ArchitectureGenome:
        """
        Breed two parent genomes into a child using HOSVD blending.

        Args:
            parent_a: First parent genome.
            parent_b: Second parent genome.
            alpha: Interpolation weight for parent_a (0 = pure B, 1 = pure A).
            genome_id: Optional ID to assign to the child.

        Returns:
            A new ArchitectureGenome.

        Raises:
            ImportError: If tensorly is not installed.
        """
        if not HAS_TENSORLY:
            raise ImportError(
                "tensorly is required for HOSVD breeding. "
                "Install it with: pip install tensorly"
            )

        tensor_a = self._weights_to_tensor(parent_a)
        tensor_b = self._weights_to_tensor(parent_b)

        # Pad to common shape before decomposition
        max_depth = max(tensor_a.shape[0], tensor_b.shape[0])
        tensor_a = self._pad_tensor(tensor_a, max_depth)
        tensor_b = self._pad_tensor(tensor_b, max_depth)

        core_a, factors_a = self._compute_hosvd(tensor_a)
        core_b, factors_b = self._compute_hosvd(tensor_b)

        # Blend in the compressed (core) space
        child_core = alpha * core_a + (1.0 - alpha) * core_b

        # Reconstruct using parent_a's factor matrices
        child_tensor = self._reconstruct(child_core, factors_a)

        return self._tensor_to_genome(child_tensor, parent_a, parent_b, genome_id=genome_id)

    # ------------------------------------------------------------------
    # Internal helpers (also exposed for unit-testing)
    # ------------------------------------------------------------------

    def _weights_to_tensor(self, genome: ArchitectureGenome) -> np.ndarray:
        """
        Convert a genome's filter counts to a 3-D tensor representation.

        Shape: (depth, 1, max_filters) — depth along axis-0,
        filter value along axis-2, singleton channel on axis-1.
        """
        depth = genome.depth
        max_f = max(genome.num_filters)
        tensor = np.zeros((depth, 1, max_f), dtype=np.float32)
        for level, f in enumerate(genome.num_filters):
            # Represent filter count as a one-hot-like row
            tensor[level, 0, :f] = 1.0
        return tensor

    def _pad_tensor(self, tensor: np.ndarray, target_depth: int) -> np.ndarray:
        """Pad tensor along axis-0 (depth) with zeros to target_depth."""
        current_depth = tensor.shape[0]
        if current_depth >= target_depth:
            return tensor
        pad_width = [(0, target_depth - current_depth)] + [(0, 0)] * (tensor.ndim - 1)
        return np.pad(tensor, pad_width, mode="constant")

    def _compute_ranks(self, tensor: np.ndarray) -> List[int]:
        """Compute Tucker ranks from rank_ratios and tensor shape."""
        ranks = []
        for i, dim in enumerate(tensor.shape):
            ratio = self.rank_ratios[i] if i < len(self.rank_ratios) else 1.0
            ranks.append(max(1, int(round(dim * ratio))))
        return ranks

    def _compute_hosvd(
        self, tensor: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply Tucker decomposition (HOSVD) to a numpy tensor.

        Returns:
            (core, factors) — core: compressed core tensor,
                              factors: list of factor matrices
        """
        tl.set_backend("numpy")
        ranks = self._compute_ranks(tensor)
        core, factors = tucker(tl.tensor(tensor), rank=ranks)
        return np.array(core), [np.array(f) for f in factors]

    def _reconstruct(
        self, core: np.ndarray, factors: List[np.ndarray]
    ) -> np.ndarray:
        """Reconstruct a full tensor from Tucker core and factor matrices."""
        tl.set_backend("numpy")
        # tl.tucker_to_tensor expects (core, factors)
        result = tl.tucker_to_tensor((tl.tensor(core), [tl.tensor(f) for f in factors]))
        return np.array(result)

    def _tensor_to_genome(
        self,
        tensor: np.ndarray,
        parent_a: ArchitectureGenome,
        parent_b: ArchitectureGenome,
        genome_id: Optional[int] = None,
    ) -> ArchitectureGenome:
        """
        Decode a reconstructed weight tensor back into an ArchitectureGenome.

        Strategy: count the number of "active" entries (> 0.5) per depth level
        to derive filter counts. Activation and skip connections are inherited
        from the higher-fitness parent.
        """
        depth = tensor.shape[0]
        num_filters = []
        for level in range(depth):
            row = tensor[level, 0, :]
            count = int(np.sum(row > 0.5))
            num_filters.append(max(1, count))

        # Inherit discrete hyperparameters from higher-fitness parent
        if (parent_a.fitness or 0.0) >= (parent_b.fitness or 0.0):
            dominant = parent_a
        else:
            dominant = parent_b

        skip_connections = list(dominant.skip_connections)
        # Adjust length to match derived depth
        while len(skip_connections) < depth:
            skip_connections.append(True)
        skip_connections = skip_connections[:depth]

        return ArchitectureGenome(
            num_filters=num_filters,
            depth=depth,
            activation=dominant.activation,
            skip_connections=skip_connections,
            genome_id=genome_id,
        )
