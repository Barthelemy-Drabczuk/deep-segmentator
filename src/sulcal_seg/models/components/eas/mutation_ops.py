"""Mutation operators for the 5-level architecture genome."""
import random
from typing import List, Optional

from sulcal_seg.models.components.eas.search_space import (
    NUM_LEVELS,
    ArchitectureGenome,
    _COL_CHANNELS,
)


def mutate(
    genome: ArchitectureGenome,
    config: dict,
    mutation_rate: float = 0.3,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Apply stochastic mutations to a genome.

    Each mutation operator fires independently with probability
    ``mutation_rate``.

    Args:
        genome: Source genome (not modified in-place).
        config: Component config dict (provides ``allowed_activations``
                and ``mutation_rate`` override if present).
        mutation_rate: Probability that each operator fires.
        rng: Optional seeded Random instance for reproducibility.

    Returns:
        A new mutated ArchitectureGenome.
    """
    if rng is None:
        rng = random.Random()

    rate = config.get("mutation_rate", mutation_rate)
    allowed = config.get(
        "allowed_activations", ["relu", "leaky_relu", "elu", "gelu"]
    )

    child = genome.clone()
    child.fitness = None

    if rng.random() < rate:
        child = change_filters(child, rng=rng)
    if rng.random() < rate:
        child = change_activation(child, allowed_activations=allowed, rng=rng)

    return child


def change_filters(
    genome: ArchitectureGenome,
    rng: Optional[random.Random] = None,
    min_channels: int = 16,
    max_channels: int = 512,
) -> ArchitectureGenome:
    """
    Scale one encoder level's channel count by a random factor.

    Picks a random level 0–4 and multiplies its channel count by one of
    {0.5, 0.75, 1.25, 1.5}, then clips to [min_channels, max_channels].

    Args:
        genome: Source genome.
        rng: Optional seeded Random instance.
        min_channels: Lower bound for channel count.
        max_channels: Upper bound for channel count.

    Returns:
        New genome with one level's channel count mutated.
    """
    if rng is None:
        rng = random.Random()

    child = genome.clone()
    child.fitness = None

    level = rng.randint(0, NUM_LEVELS - 1)
    factor = rng.choice([0.5, 0.75, 1.25, 1.5])
    old_ch = float(child.layer_configs[level, _COL_CHANNELS])
    new_ch = int(round(old_ch * factor))
    new_ch = max(min_channels, min(max_channels, new_ch))
    child.layer_configs[level, _COL_CHANNELS] = float(new_ch)
    return child


def change_activation(
    genome: ArchitectureGenome,
    allowed_activations: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Replace the activation function with a randomly chosen alternative.

    Args:
        genome: Source genome.
        allowed_activations: List of valid activation names to choose from.
        rng: Optional seeded Random instance.

    Returns:
        New genome with a (possibly different) activation function.
    """
    if rng is None:
        rng = random.Random()
    if allowed_activations is None:
        allowed_activations = ["relu", "leaky_relu", "elu", "gelu"]

    child = genome.clone()
    child.fitness = None
    child.activation = rng.choice(allowed_activations)
    return child


def tournament_select(
    population: List[ArchitectureGenome],
    tournament_size: int = 4,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Select a genome via tournament selection.

    Args:
        population: Pool of evaluated genomes (all must have fitness set).
        tournament_size: Number of contestants drawn at random.
        rng: Optional seeded Random instance.

    Returns:
        The genome with the highest fitness among the contestants.
    """
    if rng is None:
        rng = random.Random()

    contestants = rng.choices(
        population, k=min(tournament_size, len(population))
    )
    return max(contestants, key=lambda g: g.fitness or 0.0)
