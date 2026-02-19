"""Unit tests for EAS component — new 5-level genome API."""
import random

import numpy as np
import pytest

from sulcal_seg.models.components.eas.search_space import (
    NUM_LEVELS,
    NUM_PARAMS,
    ArchitectureGenome,
    SearchSpace,
)
from sulcal_seg.models.components.eas.mutation_ops import (
    change_activation,
    change_filters,
    mutate,
    tournament_select,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_configs() -> np.ndarray:
    """Return a minimal valid (5, 6) layer_configs array."""
    cfg = np.zeros((NUM_LEVELS, NUM_PARAMS), dtype=np.float32)
    for i in range(NUM_LEVELS):
        cfg[i, 0] = 64.0   # channels
        cfg[i, 1] = 2.0    # num_blocks
        cfg[i, 2] = 3.0    # kernel_size
        cfg[i, 3] = 2.0    # stride
        cfg[i, 4] = 0.1    # dropout
        cfg[i, 5] = 0.0    # skip_type_id
    return cfg


def _genome(**overrides) -> ArchitectureGenome:
    return ArchitectureGenome(
        layer_configs=overrides.get("layer_configs", _default_configs()),
        activation=overrides.get("activation", "relu"),
        normalization=overrides.get("normalization", "batch"),
        resnet_depth=overrides.get("resnet_depth", 18),
        fitness=overrides.get("fitness", None),
    )


# ---------------------------------------------------------------------------
# TestArchitectureGenome
# ---------------------------------------------------------------------------

class TestArchitectureGenome:
    def test_layer_configs_shape(self):
        g = _genome()
        assert g.layer_configs.shape == (NUM_LEVELS, NUM_PARAMS)
        assert g.layer_configs.shape == (5, 6)

    def test_fields_exist(self):
        g = _genome()
        assert hasattr(g, "activation")
        assert hasattr(g, "normalization")
        assert hasattr(g, "resnet_depth")
        assert hasattr(g, "fitness")
        assert hasattr(g, "genome_id")

    def test_default_field_values(self):
        g = _genome()
        assert g.activation == "relu"
        assert g.normalization == "batch"
        assert g.resnet_depth == 18
        assert g.fitness is None

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            ArchitectureGenome(
                layer_configs=np.zeros((3, 6), dtype=np.float32)
            )

    def test_clone_is_independent(self):
        g = _genome()
        clone = g.clone()
        clone.layer_configs[0, 0] = 9999.0
        assert g.layer_configs[0, 0] != 9999.0

    def test_clone_copies_metadata(self):
        g = _genome(activation="gelu", resnet_depth=34, fitness=0.8)
        clone = g.clone()
        assert clone.activation == "gelu"
        assert clone.resnet_depth == 34
        assert clone.fitness == pytest.approx(0.8)

    def test_to_from_dict_roundtrip(self):
        g = _genome(activation="gelu", fitness=0.75, resnet_depth=34)
        g2 = ArchitectureGenome.from_dict(g.to_dict())
        np.testing.assert_array_almost_equal(
            g2.layer_configs, g.layer_configs
        )
        assert g2.activation == g.activation
        assert g2.resnet_depth == g.resnet_depth
        assert g2.fitness == pytest.approx(g.fitness)

    def test_equality_same_configs(self):
        g1 = _genome()
        g2 = _genome()
        # Dataclass equality compares layer_configs array — numpy arrays use
        # element-wise comparison, so we compare manually.
        assert np.array_equal(g1.layer_configs, g2.layer_configs)
        assert g1.activation == g2.activation


# ---------------------------------------------------------------------------
# TestSearchSpace
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def _space(self) -> SearchSpace:
        return SearchSpace({"allowed_activations": ["relu", "leaky_relu"]})

    def test_sample_returns_genome(self):
        ss = self._space()
        g = ss.sample(rng=random.Random(0))
        assert isinstance(g, ArchitectureGenome)

    def test_sample_layer_configs_shape(self):
        ss = self._space()
        g = ss.sample(rng=random.Random(1))
        assert g.layer_configs.shape == (NUM_LEVELS, NUM_PARAMS)

    def test_sample_genome_id(self):
        ss = self._space()
        g = ss.sample(rng=random.Random(2), genome_id=7)
        assert g.genome_id == 7

    def test_sample_valid_genome(self):
        ss = self._space()
        for seed in range(10):
            g = ss.sample(rng=random.Random(seed))
            assert ss.validate(g), f"Genome failed validation (seed={seed})"

    def test_validate_accepts_valid_genome(self):
        ss = self._space()
        g = _genome()
        # Channels in [16, 512]; kernel=3 ∈ {3,5,7}; etc. — should pass
        assert ss.validate(g)

    def test_validate_rejects_too_large_channels(self):
        ss = SearchSpace({"max_channels": 64})
        g = _genome()
        g.layer_configs[0, 0] = 9999.0  # far above max
        assert not ss.validate(g)

    def test_validate_rejects_bad_kernel(self):
        ss = self._space()
        g = _genome()
        g.layer_configs[0, 2] = 4.0  # 4 is not in {3, 5, 7}
        assert not ss.validate(g)


# ---------------------------------------------------------------------------
# TestMutationOps
# ---------------------------------------------------------------------------

class TestMutationOps:
    _CFG = {
        "mutation_rate": 1.0,
        "allowed_activations": ["relu", "leaky_relu", "elu", "gelu"],
    }

    def test_change_filters_returns_genome(self):
        g = _genome()
        g2 = change_filters(g, rng=random.Random(0))
        assert isinstance(g2, ArchitectureGenome)
        assert g2.layer_configs.shape == (NUM_LEVELS, NUM_PARAMS)

    def test_change_filters_mutates_channels(self):
        """With enough seeds one level's channel count will differ."""
        g = _genome()
        any_changed = False
        for seed in range(20):
            g2 = change_filters(g, rng=random.Random(seed))
            if not np.array_equal(
                g.layer_configs[:, 0], g2.layer_configs[:, 0]
            ):
                any_changed = True
                break
        assert any_changed

    def test_change_filters_clears_fitness(self):
        g = _genome(fitness=0.9)
        g2 = change_filters(g)
        assert g2.fitness is None

    def test_change_filters_clips_to_bounds(self):
        cfg = _default_configs()
        cfg[:, 0] = 16.0  # already at minimum
        g = ArchitectureGenome(layer_configs=cfg)
        g2 = change_filters(g, rng=random.Random(0), min_channels=16)
        assert all(g2.layer_configs[:, 0] >= 16.0)

    def test_change_activation_returns_genome(self):
        g = _genome(activation="relu")
        g2 = change_activation(
            g, allowed_activations=["relu", "gelu"], rng=random.Random(0)
        )
        assert isinstance(g2, ArchitectureGenome)
        assert g2.activation in ["relu", "gelu"]

    def test_change_activation_clears_fitness(self):
        g = _genome(fitness=0.7, activation="relu")
        g2 = change_activation(
            g, allowed_activations=["relu", "gelu"], rng=random.Random(0)
        )
        assert g2.fitness is None

    def test_mutate_returns_genome(self):
        g = _genome()
        g2 = mutate(g, config=self._CFG, rng=random.Random(42))
        assert isinstance(g2, ArchitectureGenome)

    def test_mutate_clears_fitness(self):
        g = _genome(fitness=0.95)
        g2 = mutate(g, config=self._CFG, rng=random.Random(0))
        assert g2.fitness is None

    def test_tournament_select_returns_population_member(self):
        population = [
            _genome(fitness=0.9),
            _genome(fitness=0.1),
            _genome(fitness=0.5),
        ]
        winner = tournament_select(
            population, tournament_size=3, rng=random.Random(0)
        )
        # Use identity check: ndarray field makes `in` ambiguous via __eq__
        assert any(winner is g for g in population)

    def test_tournament_select_trivial_population(self):
        population = [_genome(fitness=0.9)]
        winner = tournament_select(
            population, tournament_size=5, rng=random.Random(0)
        )
        assert winner.fitness == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestHOSVDBreeder
# ---------------------------------------------------------------------------

class TestHOSVDBreeder:
    """Tests for HOSVDBreeder — skipped if tensorly is not installed."""

    @pytest.fixture
    def setup(self):
        from sulcal_seg.models.components.eas.hosvd_breeder import (
            HAS_TENSORLY,
            HOSVDBreeder,
        )
        if not HAS_TENSORLY:
            pytest.skip("tensorly not installed")
        breeder = HOSVDBreeder(rank_ratios=(0.8, 0.8))
        pa = _genome(activation="relu", fitness=0.8)
        pb = _genome(activation="leaky_relu", fitness=0.6)
        # Give parents slightly different configs so breeding is non-trivial
        pb.layer_configs[:, 0] = 128.0
        return breeder, pa, pb

    def test_genome_to_tensor_shape(self, setup):
        breeder, pa, _ = setup
        t = breeder._genome_to_tensor(pa)
        assert t.shape == (NUM_LEVELS, NUM_PARAMS)

    def test_genome_to_tensor_dtype(self, setup):
        breeder, pa, _ = setup
        t = breeder._genome_to_tensor(pa)
        assert t.dtype == np.float32

    def test_tensor_to_genome_roundtrip(self, setup):
        breeder, pa, _ = setup
        t = breeder._genome_to_tensor(pa)
        g2 = breeder._tensor_to_genome(t, "relu", "batch", 18)
        assert isinstance(g2, ArchitectureGenome)
        assert g2.layer_configs.shape == (NUM_LEVELS, NUM_PARAMS)
        # Channels should survive the round-trip (clipped, but 64 is valid)
        for level in range(NUM_LEVELS):
            ch = int(g2.layer_configs[level, 0])
            assert 16 <= ch <= 512

    def test_breed_offspring_returns_genome(self, setup):
        breeder, pa, pb = setup
        child = breeder.breed_offspring(pa, pb, genome_id=99)
        assert isinstance(child, ArchitectureGenome)
        assert child.layer_configs.shape == (NUM_LEVELS, NUM_PARAMS)
        assert child.genome_id == 99
