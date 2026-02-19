"""EAS (Evolutionary Architecture Search) engine — main component."""
import random
from typing import Any, Dict, List, Optional

from sulcal_seg.models.components.base_component import BaseComponent
from sulcal_seg.models.components.eas.fitness_eval import FitnessEvaluator
from sulcal_seg.models.components.eas.hosvd_breeder import HOSVDBreeder
from sulcal_seg.models.components.eas.mutation_ops import mutate, tournament_select
from sulcal_seg.models.components.eas.search_space import ArchitectureGenome, SearchSpace


_REQUIRED_KEYS = frozenset({"population_size", "num_generations", "num_classes"})


class EASEngine(BaseComponent):
    """
    Evolutionary Architecture Search engine.

    Evolves a population of U-Net architectures over multiple generations
    using HOSVD-based crossover and random mutation operators.

    The full training loop (`_evolve_population`, `forward`) requires GPU and
    real data loaders — they are marked as NotImplementedError stubs.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.search_space = SearchSpace(config)
        self.breeder = HOSVDBreeder(
            rank_ratios=tuple(config.get("hosvd_rank_ratios", (0.8, 0.8)))
        )
        self.evaluator = FitnessEvaluator(config)
        self.population: List[ArchitectureGenome] = []
        self.best_genome: Optional[ArchitectureGenome] = None
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # BaseComponent interface
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = _REQUIRED_KEYS - set(self.config.keys())
        if missing:
            raise ValueError(f"EASEngine config missing keys: {missing}")
        if self.config["population_size"] < 4:
            raise ValueError("population_size must be >= 4")
        if self.config["num_classes"] < 2:
            raise ValueError("num_classes must be >= 2")

    def train(self, train_loader: Any, val_loader: Any) -> Dict[str, float]:
        """
        Run the evolutionary search loop.

        Args:
            train_loader: Training DataLoader passed to FitnessEvaluator.
            val_loader: Validation DataLoader passed to FitnessEvaluator.

        Returns:
            Dict with 'best_val_dice' of the discovered architecture.

        Raises:
            NotImplementedError: Always (stub — requires GPU + data).
        """
        # TODO: Call self._initialize_population(), then for each generation:
        #   1. Evaluate unevaluated genomes via self.evaluator.evaluate()
        #   2. Select elite, breed offspring via self.breeder.breed_offspring()
        #   3. Apply mutations via mutate()
        #   4. Replace weakest individuals
        #   5. Track self.best_genome
        raise NotImplementedError(
            "EASEngine.train() requires GPU and real data loaders. "
            "Implement the generational loop here."
        )

    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Run the best discovered architecture's forward pass.

        Raises:
            NotImplementedError: Until a best_genome with trained weights exists.
        """
        raise NotImplementedError(
            "EASEngine.forward() is only available after a completed evolutionary search "
            "and requires loading the best architecture's trained weights."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_population(self) -> List[ArchitectureGenome]:
        """
        Sample `population_size` random architectures from the search space.

        Returns:
            List of freshly sampled genomes (all fitness=None).
        """
        n = self.config["population_size"]
        self.population = [
            self.search_space.sample(genome_id=i) for i in range(n)
        ]
        return self.population

    def _evolve_population(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> List[ArchitectureGenome]:
        """
        Run one generation: evaluate → select → breed → mutate.

        Raises:
            NotImplementedError: Always (stub — requires GPU + data).
        """
        raise NotImplementedError(
            "EASEngine._evolve_population() requires GPU and real data loaders."
        )

    def _get_elite(self, fraction: float = 0.2) -> List[ArchitectureGenome]:
        """Return top `fraction` of the current population by fitness."""
        evaluated = [g for g in self.population if g.fitness is not None]
        evaluated.sort(key=lambda g: g.fitness or 0.0, reverse=True)
        n_elite = max(1, int(len(evaluated) * fraction))
        return evaluated[:n_elite]
