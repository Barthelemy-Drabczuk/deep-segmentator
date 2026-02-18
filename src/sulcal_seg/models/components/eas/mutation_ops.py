"""Mutation operators for architecture genomes."""
import random
from typing import List, Optional

from sulcal_seg.models.components.eas.search_space import ArchitectureGenome, SearchSpace


def mutate(
    genome: ArchitectureGenome,
    search_space: SearchSpace,
    mutation_rate: float = 0.3,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Apply stochastic mutations to a genome.

    Each mutation operator fires independently with probability `mutation_rate`.

    Args:
        genome: Source genome (not modified in-place).
        search_space: Defines valid ranges; child will be clamped.
        mutation_rate: Probability that each mutation fires.
        rng: Optional seeded Random instance for reproducibility.

    Returns:
        A new mutated ArchitectureGenome.
    """
    if rng is None:
        rng = random.Random()

    child = genome.clone()
    child.fitness = None  # child must be re-evaluated

    if rng.random() < mutation_rate:
        child = change_filters(child, search_space, rng=rng)
    if rng.random() < mutation_rate:
        child = change_activation(child, search_space, rng=rng)
    if rng.random() < mutation_rate:
        child = change_depth(child, search_space, rng=rng)

    return search_space.clamp(child)


def change_filters(
    genome: ArchitectureGenome,
    search_space: SearchSpace,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Randomly scale one encoder level's filter count by ×0.5, ×1, or ×2.

    Args:
        genome: Source genome.
        search_space: Used to validate/clamp the result.
        rng: Optional seeded Random instance.

    Returns:
        New genome with one level's filter count mutated.
    """
    if rng is None:
        rng = random.Random()

    child = genome.clone()
    child.fitness = None

    level = rng.randint(0, child.depth - 1)
    factor = rng.choice([0.5, 2.0])
    new_f = max(
        search_space.min_filters,
        min(search_space.max_filters, int(child.num_filters[level] * factor)),
    )
    child.num_filters[level] = new_f
    return child


def change_activation(
    genome: ArchitectureGenome,
    search_space: SearchSpace,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Replace the activation function with a randomly chosen alternative.

    Args:
        genome: Source genome.
        search_space: Provides the list of allowed activations.
        rng: Optional seeded Random instance.

    Returns:
        New genome with a different activation function.
    """
    if rng is None:
        rng = random.Random()

    child = genome.clone()
    child.fitness = None

    alternatives = [a for a in search_space.allowed_activations if a != genome.activation]
    if alternatives:
        child.activation = rng.choice(alternatives)
    return child


def change_depth(
    genome: ArchitectureGenome,
    search_space: SearchSpace,
    rng: Optional[random.Random] = None,
) -> ArchitectureGenome:
    """
    Add or remove one encoder level (±1 depth), respecting search-space bounds.

    When adding a level: duplicate the last filter count.
    When removing a level: drop the last encoder level.

    Args:
        genome: Source genome.
        search_space: Defines min_depth / max_depth bounds.
        rng: Optional seeded Random instance.

    Returns:
        New genome with depth changed by ±1.
    """
    if rng is None:
        rng = random.Random()

    child = genome.clone()
    child.fitness = None

    can_grow = child.depth < search_space.max_depth
    can_shrink = child.depth > search_space.min_depth

    if can_grow and can_shrink:
        grow = rng.random() < 0.5
    elif can_grow:
        grow = True
    elif can_shrink:
        grow = False
    else:
        return child  # stuck at boundary, no-op

    if grow:
        # Add a new deepest level: duplicate last filter count
        child.num_filters.append(child.num_filters[-1])
        child.skip_connections.append(True)
        child.depth += 1
    else:
        # Remove deepest level
        child.num_filters.pop()
        child.skip_connections.pop()
        child.depth -= 1

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
        tournament_size: Number of contestants.
        rng: Optional seeded Random instance.

    Returns:
        The genome with the highest fitness among the random contestants.
    """
    if rng is None:
        rng = random.Random()

    contestants = rng.choices(population, k=min(tournament_size, len(population)))
    return max(contestants, key=lambda g: g.fitness or 0.0)
