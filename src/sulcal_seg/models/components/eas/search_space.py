"""Architecture genome and search space for EAS (NAS with HOSVD breeding)."""
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LEVELS: int = 5
NUM_PARAMS: int = 6

# Column indices for layer_configs array
_COL_CHANNELS = 0
_COL_NUM_BLOCKS = 1
_COL_KERNEL_SIZE = 2
_COL_STRIDE = 3
_COL_DROPOUT = 4
_COL_SKIP_TYPE_ID = 5

_VALID_KERNEL_SIZES = (3, 5, 7)
_VALID_SKIP_TYPES = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# ArchitectureGenome
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureGenome:
    """
    Fixed 5-level architecture genome for the NAS search.

    Each of the 5 encoder levels is described by 6 numerical parameters
    stored as a row in ``layer_configs``:

    Col 0 — channels        : int  in [16, 512]
    Col 1 — num_blocks      : int  in [1, 6]
    Col 2 — kernel_size     : int  in {3, 5, 7}
    Col 3 — stride          : int  in {1, 2}
    Col 4 — dropout         : float in [0.0, 0.5]
    Col 5 — skip_type_id    : int  in {0, 1, 2, 3}

    Attributes:
        layer_configs: Float32 array of shape (NUM_LEVELS, NUM_PARAMS).
        activation: Activation function name (e.g. 'relu', 'gelu', 'mish').
        normalization: Normalisation type ('batch', 'group', 'layer').
        resnet_depth: ResNet backbone variant (18, 34, or 50).
        fitness: Validation metric after evaluation (None = unevaluated).
        genome_id: Optional unique identifier.
    """

    layer_configs: np.ndarray   # shape (5, 6), dtype float32
    activation: str = "relu"
    normalization: str = "batch"
    resnet_depth: int = 18
    fitness: Optional[float] = field(default=None, compare=False)
    genome_id: Optional[int] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.layer_configs = np.asarray(self.layer_configs, dtype=np.float32)
        if self.layer_configs.shape != (NUM_LEVELS, NUM_PARAMS):
            raise ValueError(
                f"layer_configs must have shape ({NUM_LEVELS}, {NUM_PARAMS}), "
                f"got {self.layer_configs.shape}"
            )

    def clone(self) -> "ArchitectureGenome":
        return ArchitectureGenome(
            layer_configs=self.layer_configs.copy(),
            activation=self.activation,
            normalization=self.normalization,
            resnet_depth=self.resnet_depth,
            fitness=self.fitness,
            genome_id=self.genome_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_configs": self.layer_configs.tolist(),
            "activation": self.activation,
            "normalization": self.normalization,
            "resnet_depth": self.resnet_depth,
            "fitness": self.fitness,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureGenome":
        return cls(
            layer_configs=np.array(d["layer_configs"], dtype=np.float32),
            activation=d.get("activation", "relu"),
            normalization=d.get("normalization", "batch"),
            resnet_depth=d.get("resnet_depth", 18),
            fitness=d.get("fitness"),
            genome_id=d.get("genome_id"),
        )


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


class SearchSpace:
    """
    Defines valid parameter ranges and samples random genomes.

    Accepts the full component config dict; extra keys are ignored so that
    old configs (min_filters, max_filters, min_depth, max_depth) remain
    valid without any changes to existing test fixtures.

    Args:
        config: Component config dict (or kwargs via ``**config``).
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        cfg = dict(config or {})
        cfg.update(kwargs)

        self.min_channels: int = cfg.get("min_channels", 16)
        self.max_channels: int = cfg.get("max_channels", 512)
        self.allowed_activations: List[str] = cfg.get(
            "allowed_activations", ["relu", "leaky_relu", "elu", "gelu"]
        )
        self.allowed_normalizations: List[str] = cfg.get(
            "allowed_normalizations", ["batch", "group", "layer"]
        )
        self.allowed_resnet_depths: List[int] = cfg.get(
            "allowed_resnet_depths", [18, 34, 50]
        )

    def sample(
        self,
        rng: Optional[random.Random] = None,
        genome_id: Optional[int] = None,
    ) -> ArchitectureGenome:
        """
        Sample a random genome from the search space.

        Args:
            rng: Optional seeded Random instance for reproducibility.
            genome_id: Optional ID to assign to the genome.

        Returns:
            A randomly initialised ArchitectureGenome.
        """
        if rng is None:
            rng = random.Random()

        configs = np.zeros((NUM_LEVELS, NUM_PARAMS), dtype=np.float32)
        for level in range(NUM_LEVELS):
            # Channels roughly double each level (common U-Net pattern)
            base = rng.randint(self.min_channels, max(self.min_channels, 64))
            channels = min(self.max_channels, base * (2 ** level))
            channels = max(self.min_channels, channels)
            configs[level, _COL_CHANNELS] = float(channels)
            configs[level, _COL_NUM_BLOCKS] = float(rng.randint(1, 4))
            ks = float(rng.choice(_VALID_KERNEL_SIZES))
            configs[level, _COL_KERNEL_SIZE] = ks
            # enc0 always stride-1 (no downsampling); enc1–enc4 stride-2
            configs[level, _COL_STRIDE] = 1.0 if level == 0 else 2.0
            configs[level, _COL_DROPOUT] = round(rng.uniform(0.0, 0.3), 2)
            skip = float(rng.choice(_VALID_SKIP_TYPES))
            configs[level, _COL_SKIP_TYPE_ID] = skip

        activation = rng.choice(self.allowed_activations)
        normalization = rng.choice(self.allowed_normalizations)
        resnet_depth = rng.choice(self.allowed_resnet_depths)

        return ArchitectureGenome(
            layer_configs=configs,
            activation=activation,
            normalization=normalization,
            resnet_depth=resnet_depth,
            genome_id=genome_id,
        )

    def validate(self, genome: ArchitectureGenome) -> bool:
        """Return True if the genome lies within valid parameter bounds."""
        if genome.layer_configs.shape != (NUM_LEVELS, NUM_PARAMS):
            return False
        for level in range(NUM_LEVELS):
            ch = int(genome.layer_configs[level, _COL_CHANNELS])
            if not (self.min_channels <= ch <= self.max_channels):
                return False
            nb = int(round(genome.layer_configs[level, _COL_NUM_BLOCKS]))
            if not (1 <= nb <= 6):
                return False
            ks = int(round(genome.layer_configs[level, _COL_KERNEL_SIZE]))
            if ks not in _VALID_KERNEL_SIZES:
                return False
            stride = int(round(genome.layer_configs[level, _COL_STRIDE]))
            if stride not in (1, 2):
                return False
            dropout = float(genome.layer_configs[level, _COL_DROPOUT])
            if not (0.0 <= dropout <= 0.5):
                return False
            skip = int(round(genome.layer_configs[level, _COL_SKIP_TYPE_ID]))
            if skip not in _VALID_SKIP_TYPES:
                return False
        if genome.activation not in self.allowed_activations:
            return False
        return True
