"""Architecture genome and search space for EAS."""
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArchitectureGenome:
    """
    Represents a single candidate U-Net architecture.

    Attributes:
        num_filters: List of filter counts per encoder level
        depth: Number of encoder/decoder levels
        activation: Activation function name
        skip_connections: Whether each level uses skip connections
        fitness: Validation metric assigned after evaluation (None = not yet evaluated)
        genome_id: Optional unique identifier
    """

    num_filters: List[int]
    depth: int
    activation: str
    skip_connections: List[bool]
    fitness: Optional[float] = field(default=None, compare=False)
    genome_id: Optional[int] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if len(self.num_filters) != self.depth:
            raise ValueError(
                f"len(num_filters)={len(self.num_filters)} must equal depth={self.depth}"
            )
        if len(self.skip_connections) != self.depth:
            raise ValueError(
                f"len(skip_connections)={len(self.skip_connections)} must equal depth={self.depth}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise genome to a plain dict (for logging / checkpointing)."""
        return {
            "num_filters": self.num_filters,
            "depth": self.depth,
            "activation": self.activation,
            "skip_connections": self.skip_connections,
            "fitness": self.fitness,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureGenome":
        return cls(
            num_filters=d["num_filters"],
            depth=d["depth"],
            activation=d["activation"],
            skip_connections=d["skip_connections"],
            fitness=d.get("fitness"),
            genome_id=d.get("genome_id"),
        )

    def clone(self) -> "ArchitectureGenome":
        return ArchitectureGenome(
            num_filters=list(self.num_filters),
            depth=self.depth,
            activation=self.activation,
            skip_connections=list(self.skip_connections),
            fitness=self.fitness,
            genome_id=self.genome_id,
        )


class SearchSpace:
    """
    Defines the valid range of architectures and samples random genomes.

    Attributes:
        min_filters: Minimum number of filters at any level
        max_filters: Maximum number of filters at any level
        min_depth: Minimum U-Net depth (encoder levels)
        max_depth: Maximum U-Net depth (encoder levels)
        allowed_activations: List of allowed activation function names
    """

    def __init__(
        self,
        min_filters: int = 8,
        max_filters: int = 256,
        min_depth: int = 2,
        max_depth: int = 6,
        allowed_activations: Optional[List[str]] = None,
    ) -> None:
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.allowed_activations = allowed_activations or [
            "relu", "leaky_relu", "elu", "gelu"
        ]

    def sample(self, genome_id: Optional[int] = None) -> ArchitectureGenome:
        """
        Sample a random architecture genome from the search space.

        Returns:
            A randomly initialised ArchitectureGenome.
        """
        depth = random.randint(self.min_depth, self.max_depth)
        # Filters roughly double each level (common U-Net pattern), clipped to bounds
        base = random.randint(self.min_filters, max(self.min_filters, self.max_filters // 8))
        num_filters = [
            min(self.max_filters, max(self.min_filters, base * (2 ** level)))
            for level in range(depth)
        ]
        activation = random.choice(self.allowed_activations)
        skip_connections = [True] * depth  # always start with skip connections

        return ArchitectureGenome(
            num_filters=num_filters,
            depth=depth,
            activation=activation,
            skip_connections=skip_connections,
            genome_id=genome_id,
        )

    def is_valid(self, genome: ArchitectureGenome) -> bool:
        """Check whether a genome lies within the search space bounds."""
        if not (self.min_depth <= genome.depth <= self.max_depth):
            return False
        if len(genome.num_filters) != genome.depth:
            return False
        for f in genome.num_filters:
            if not (self.min_filters <= f <= self.max_filters):
                return False
        if genome.activation not in self.allowed_activations:
            return False
        return True

    def clamp(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Return a copy of genome with all values clamped to valid bounds."""
        g = genome.clone()
        g.depth = max(self.min_depth, min(self.max_depth, g.depth))
        # Adjust filter list to match depth
        while len(g.num_filters) < g.depth:
            g.num_filters.append(g.num_filters[-1] if g.num_filters else self.min_filters)
        g.num_filters = [
            max(self.min_filters, min(self.max_filters, f)) for f in g.num_filters[: g.depth]
        ]
        while len(g.skip_connections) < g.depth:
            g.skip_connections.append(True)
        g.skip_connections = g.skip_connections[: g.depth]
        if g.activation not in self.allowed_activations:
            g.activation = self.allowed_activations[0]
        return g
