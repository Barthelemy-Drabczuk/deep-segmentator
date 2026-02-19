"""HOSVD-based architecture breeding operator for EAS.

Tucker decomposition (Higher-Order SVD) blends two parent architecture
tensors of shape (NUM_LEVELS, NUM_PARAMS) = (5, 6) in a compressed
low-rank space, then reconstructs an offspring genome.

Requires: tensorly (optional — guarded by HAS_TENSORLY flag).
If tensorly is not installed, breed_offspring() raises ImportError.
"""
from typing import List, Optional, Tuple

import numpy as np

from sulcal_seg.models.components.eas.search_space import (
    NUM_LEVELS,
    NUM_PARAMS,
    ArchitectureGenome,
    _VALID_KERNEL_SIZES,
    _VALID_SKIP_TYPES,
)

try:
    import tensorly as tl
    from tensorly.decomposition import tucker

    HAS_TENSORLY = True
except ImportError:  # pragma: no cover
    HAS_TENSORLY = False


class HOSVDBreeder:
    """
    Breeds two parent ArchitectureGenomes via HOSVD-guided tensor blending.

    Breeding process:
    1. Encode each parent's layer_configs as a (NUM_LEVELS, NUM_PARAMS) tensor
    2. Apply Tucker decomposition to extract core + factor matrices
    3. Blend parent cores: child_core = 0.6*core_a + 0.4*core_b
    4. Reconstruct child tensor using parent_a's factor matrices
    5. Decode child tensor back to a valid ArchitectureGenome

    Args:
        rank_ratios: Fraction of each mode's dimension to retain in Tucker
                     decomposition. Two values — one per tensor dimension
                     (NUM_LEVELS, NUM_PARAMS). If a list of 3+ values is
                     provided (e.g. from a legacy config), only the first
                     two are used.
    """

    def __init__(
        self, rank_ratios: Tuple[float, ...] = (0.8, 0.8)
    ) -> None:
        # Use only first two values to support legacy 3-element configs
        r = tuple(rank_ratios)
        self.rank_ratios: Tuple[float, float] = (
            float(r[0]) if len(r) > 0 else 0.8,
            float(r[1]) if len(r) > 1 else 0.8,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def breed_offspring(
        self,
        parent_a: ArchitectureGenome,
        parent_b: ArchitectureGenome,
        genome_id: Optional[int] = None,
    ) -> ArchitectureGenome:
        """
        Breed two parent genomes into a child using HOSVD blending.

        Args:
            parent_a: First parent genome (core weighted 0.6).
            parent_b: Second parent genome (core weighted 0.4).
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

        tensor_a = self._genome_to_tensor(parent_a)
        tensor_b = self._genome_to_tensor(parent_b)

        core_a, factors_a = self._compute_hosvd(tensor_a)
        core_b, _ = self._compute_hosvd(tensor_b)

        # Blend cores; reconstruct with parent_a's factor matrices
        child_core = 0.6 * core_a + 0.4 * core_b
        child_tensor = self._reconstruct(child_core, factors_a)

        # Inherit scalar hyperparams from the higher-fitness parent
        dominant = (
            parent_a
            if (parent_a.fitness or 0.0) >= (parent_b.fitness or 0.0)
            else parent_b
        )
        return self._tensor_to_genome(
            child_tensor,
            activation=dominant.activation,
            normalization=dominant.normalization,
            resnet_depth=dominant.resnet_depth,
            genome_id=genome_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers (exposed for unit testing)
    # ------------------------------------------------------------------

    def _genome_to_tensor(self, genome: ArchitectureGenome) -> np.ndarray:
        """
        Return the genome's layer_configs as a (NUM_LEVELS, NUM_PARAMS)
        float32 array (the tensor used in Tucker decomposition).
        """
        return genome.layer_configs.copy()

    def _tensor_to_genome(
        self,
        tensor: np.ndarray,
        activation: str = "relu",
        normalization: str = "batch",
        resnet_depth: int = 18,
        genome_id: Optional[int] = None,
    ) -> ArchitectureGenome:
        """
        Decode a (NUM_LEVELS, NUM_PARAMS) tensor back to an ArchitectureGenome.

        Each column is clipped / rounded to the nearest valid discrete value.
        """
        configs = np.zeros((NUM_LEVELS, NUM_PARAMS), dtype=np.float32)
        for level in range(NUM_LEVELS):
            row = tensor[level]

            # channels: int in [16, 512]
            ch = int(np.clip(round(float(row[0])), 16, 512))
            configs[level, 0] = float(ch)

            # num_blocks: int in [1, 6]
            nb = int(np.clip(round(float(row[1])), 1, 6))
            configs[level, 1] = float(nb)

            # kernel_size: nearest of {3, 5, 7}
            ks_raw = float(row[2])
            ks = min(_VALID_KERNEL_SIZES, key=lambda k: abs(k - ks_raw))
            configs[level, 2] = float(ks)

            # stride: nearest of {1, 2}
            v = float(row[3])
            stride = 1 if abs(v - 1.0) < abs(v - 2.0) else 2
            configs[level, 3] = float(stride)

            # dropout: float in [0.0, 0.5]
            dropout = float(np.clip(row[4], 0.0, 0.5))
            configs[level, 4] = round(dropout, 3)

            # skip_type_id: nearest of {0, 1, 2, 3}
            skip_raw = float(row[5])
            skip = min(_VALID_SKIP_TYPES, key=lambda s: abs(s - skip_raw))
            configs[level, 5] = float(skip)

        return ArchitectureGenome(
            layer_configs=configs,
            activation=activation,
            normalization=normalization,
            resnet_depth=resnet_depth,
            genome_id=genome_id,
        )

    def _compute_ranks(self, tensor: np.ndarray) -> List[int]:
        """Compute Tucker ranks from rank_ratios and tensor shape."""
        return [
            max(1, int(round(tensor.shape[0] * self.rank_ratios[0]))),
            max(1, int(round(tensor.shape[1] * self.rank_ratios[1]))),
        ]

    def _compute_hosvd(
        self, tensor: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply Tucker decomposition (HOSVD) to a (NUM_LEVELS, NUM_PARAMS)
        tensor.

        Returns:
            (core, factors) — core: compressed core tensor,
                              factors: list of two factor matrices.
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
        result = tl.tucker_to_tensor(
            (tl.tensor(core), [tl.tensor(f) for f in factors])
        )
        return np.array(result)
