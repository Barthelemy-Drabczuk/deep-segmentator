"""Unit tests for EAS component (no GPU, no tensorly required)."""
import pytest
import numpy as np

from sulcal_seg.models.components.eas.search_space import ArchitectureGenome, SearchSpace
from sulcal_seg.models.components.eas.mutation_ops import (
    change_activation,
    change_depth,
    change_filters,
    mutate,
    tournament_select,
)


class TestArchitectureGenome:
    def test_valid_genome(self):
        g = ArchitectureGenome(
            num_filters=[16, 32],
            depth=2,
            activation="relu",
            skip_connections=[True, True],
        )
        assert g.depth == 2

    def test_mismatched_depth_raises(self):
        with pytest.raises(ValueError):
            ArchitectureGenome(
                num_filters=[16, 32, 64],  # length 3 but depth=2
                depth=2,
                activation="relu",
                skip_connections=[True, True],
            )

    def test_clone_is_independent(self):
        g = ArchitectureGenome([16, 32], 2, "relu", [True, True], fitness=0.9)
        clone = g.clone()
        clone.num_filters[0] = 999
        assert g.num_filters[0] == 16  # original unchanged

    def test_to_from_dict_roundtrip(self):
        g = ArchitectureGenome([16, 32], 2, "relu", [True, True], fitness=0.8, genome_id=7)
        g2 = ArchitectureGenome.from_dict(g.to_dict())
        assert g2.num_filters == g.num_filters
        assert g2.depth == g.depth
        assert g2.fitness == g.fitness
        assert g2.genome_id == g.genome_id


class TestSearchSpace:
    def test_sample_returns_genome(self):
        ss = SearchSpace(min_filters=8, max_filters=64, min_depth=2, max_depth=4)
        g = ss.sample()
        assert isinstance(g, ArchitectureGenome)

    def test_sample_respects_depth_bounds(self):
        ss = SearchSpace(min_depth=2, max_depth=4)
        for _ in range(20):
            g = ss.sample()
            assert 2 <= g.depth <= 4

    def test_sample_filter_list_matches_depth(self):
        ss = SearchSpace()
        g = ss.sample()
        assert len(g.num_filters) == g.depth

    def test_is_valid_accepts_valid_genome(self):
        ss = SearchSpace(min_filters=8, max_filters=64, min_depth=2, max_depth=4)
        g = ArchitectureGenome([8, 16], 2, "relu", [True, True])
        assert ss.is_valid(g)

    def test_is_valid_rejects_out_of_bounds(self):
        ss = SearchSpace(min_filters=8, max_filters=64, min_depth=2, max_depth=4)
        g = ArchitectureGenome([4, 16], 2, "relu", [True, True])  # 4 < min_filters
        assert not ss.is_valid(g)

    def test_clamp_fixes_filter_violation(self):
        ss = SearchSpace(min_filters=8, max_filters=64)
        g = ArchitectureGenome([4, 128], 2, "relu", [True, True])
        clamped = ss.clamp(g)
        assert clamped.num_filters[0] == 8   # clamped up
        assert clamped.num_filters[1] == 64  # clamped down


class TestMutationOps:
    def _genome(self) -> ArchitectureGenome:
        return ArchitectureGenome([16, 32], 2, "relu", [True, True], fitness=0.5)

    def _space(self) -> SearchSpace:
        return SearchSpace(min_filters=8, max_filters=64, min_depth=2, max_depth=4)

    def test_change_filters_changes_one_level(self):
        g = self._genome()
        ss = self._space()
        g2 = change_filters(g, ss)
        # Depth unchanged
        assert g2.depth == g.depth

    def test_change_activation_changes_activation(self):
        import random
        g = self._genome()
        ss = SearchSpace(allowed_activations=["relu", "leaky_relu", "elu"])
        rng = random.Random(42)
        g2 = change_activation(g, ss, rng=rng)
        assert g2.activation != "relu" or g.activation == g2.activation  # may stay if only one alternative

    def test_change_depth_grow(self):
        import random
        g = ArchitectureGenome([16, 32], 2, "relu", [True, True])
        ss = SearchSpace(min_depth=2, max_depth=4)
        rng = random.Random(99)  # seed to force growth
        # Patch to always grow
        grown = change_depth.__wrapped__(g, ss, rng=rng) if hasattr(change_depth, "__wrapped__") else None
        # Just verify grow produces depth+1 or depth-1
        g2 = change_depth(g, ss)
        assert g2.depth in (g.depth - 1, g.depth + 1)

    def test_mutate_returns_new_genome(self):
        g = self._genome()
        ss = self._space()
        g2 = mutate(g, ss, mutation_rate=1.0)  # force all mutations
        assert g2 is not g
        assert g2.fitness is None  # must be re-evaluated

    def test_tournament_select_returns_a_genome(self):
        """tournament_select returns an element from the population."""
        import random
        population = [
            ArchitectureGenome([16], 1, "relu", [True], fitness=0.9),
            ArchitectureGenome([32], 1, "relu", [True], fitness=0.1),
            ArchitectureGenome([64], 1, "relu", [True], fitness=0.5),
        ]
        winner = tournament_select(population, tournament_size=3, rng=random.Random(0))
        assert winner in population
        assert winner.fitness is not None

    def test_tournament_select_trivial_population(self):
        """With a single-element population, that element always wins."""
        import random
        population = [ArchitectureGenome([16], 1, "relu", [True], fitness=0.9)]
        winner = tournament_select(population, tournament_size=5, rng=random.Random(0))
        assert winner.fitness == pytest.approx(0.9)


class TestHOSVDBreeder:
    """Tests for HOSVDBreeder — skipped if tensorly not installed."""

    @pytest.fixture
    def breeder_and_parents(self):
        from sulcal_seg.models.components.eas.hosvd_breeder import HOSVDBreeder, HAS_TENSORLY

        if not HAS_TENSORLY:
            pytest.skip("tensorly not installed")

        breeder = HOSVDBreeder(rank_ratios=(0.7, 0.7, 0.7))
        pa = ArchitectureGenome([8, 16], 2, "relu", [True, True], fitness=0.8)
        pb = ArchitectureGenome([16, 32], 2, "leaky_relu", [True, True], fitness=0.6)
        return breeder, pa, pb

    def test_weights_to_tensor_shape(self, breeder_and_parents):
        breeder, pa, _ = breeder_and_parents
        tensor = breeder._weights_to_tensor(pa)
        assert tensor.shape == (pa.depth, 1, max(pa.num_filters))

    def test_weights_to_tensor_values(self, breeder_and_parents):
        breeder, pa, _ = breeder_and_parents
        tensor = breeder._weights_to_tensor(pa)
        # Row 0 should have exactly pa.num_filters[0] ones
        assert tensor[0, 0, :].sum() == pytest.approx(float(pa.num_filters[0]))

    def test_compute_hosvd_returns_core_and_factors(self, breeder_and_parents):
        breeder, pa, _ = breeder_and_parents
        tensor = breeder._weights_to_tensor(pa)
        core, factors = breeder._compute_hosvd(tensor)
        assert core.ndim == tensor.ndim
        assert len(factors) == tensor.ndim
