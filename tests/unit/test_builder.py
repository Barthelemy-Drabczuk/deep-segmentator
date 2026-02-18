"""Unit tests for ModelBuilder — verification target.

All tests must pass without GPU or real data.
Run: pytest tests/unit/test_builder.py -v
"""
import pytest

from sulcal_seg.models.builder import ModelBuilder
from sulcal_seg.models.full_model import SulcalSegmentationModel
from tests.fixtures.mock_configs import make_eas_config, make_gan_config, make_laplace_config


# ---------------------------------------------------------------------------
# Baseline (no components)
# ---------------------------------------------------------------------------

class TestBaselineModel:
    def test_build_baseline(self):
        """ModelBuilder().build() produces a valid model with no components."""
        model = ModelBuilder().build()
        assert isinstance(model, SulcalSegmentationModel)
        assert not model.has_eas
        assert not model.has_gan
        assert not model.has_laplace

    def test_baseline_repr(self):
        model = ModelBuilder().build()
        assert "baseline" in repr(model)

    def test_baseline_component_summary(self):
        model = ModelBuilder().build()
        summary = model.component_summary()
        assert summary["has_eas"] is False
        assert summary["has_gan"] is False
        assert summary["has_laplace"] is False
        assert summary["active_components"] == []


# ---------------------------------------------------------------------------
# Single-component models
# ---------------------------------------------------------------------------

class TestEASOnlyModel:
    def test_with_eas_sets_flag(self):
        model = ModelBuilder().with_eas(make_eas_config()).build()
        assert model.has_eas is True
        assert model.has_gan is False
        assert model.has_laplace is False

    def test_with_eas_creates_engine(self):
        from sulcal_seg.models.components.eas.eas_engine import EASEngine

        model = ModelBuilder().with_eas(make_eas_config()).build()
        assert isinstance(model.eas, EASEngine)

    def test_eas_population_initialized_lazily(self):
        """EASEngine can be instantiated — no training needed."""
        model = ModelBuilder().with_eas(make_eas_config()).build()
        # Just check the engine exists and config is stored
        assert model.eas.config["num_classes"] == make_eas_config()["num_classes"]


class TestGANOnlyModel:
    def test_with_gan_sets_flag(self):
        model = ModelBuilder().with_gan(make_gan_config()).build()
        assert model.has_gan is True
        assert model.has_eas is False
        assert model.has_laplace is False

    def test_with_gan_creates_generator_and_discriminator(self):
        import torch.nn as nn

        model = ModelBuilder().with_gan(make_gan_config()).build()
        assert model.gan is not None
        assert isinstance(model.gan.generator, nn.Module)
        assert isinstance(model.gan.discriminator, nn.Module)

    def test_gan_generator_output_channels(self):
        import torch

        cfg = make_gan_config()
        model = ModelBuilder().with_gan(cfg).build()
        x = torch.zeros(1, cfg["in_channels"], 8, 8, 8)
        out = model.gan.generator(x)
        assert out.shape[1] == cfg["num_classes"]

    def test_gan_discriminator_output_shape(self):
        import torch

        cfg = make_gan_config()
        model = ModelBuilder().with_gan(cfg).build()
        B, C = 1, cfg["num_classes"]
        image = torch.zeros(B, 1, 8, 8, 8)
        seg = torch.zeros(B, C, 8, 8, 8)
        out = model.gan.discriminator(image, seg)
        assert out.shape == (B, 1)


class TestLaplaceOnlyModel:
    def test_with_laplace_sets_flag(self):
        model = ModelBuilder().with_laplace(make_laplace_config()).build()
        assert model.has_laplace is True
        assert model.has_eas is False
        assert model.has_gan is False

    def test_with_laplace_creates_engine(self):
        from sulcal_seg.models.components.laplace.laplace_engine import LaplaceEngine

        model = ModelBuilder().with_laplace(make_laplace_config()).build()
        assert isinstance(model.laplace, LaplaceEngine)


# ---------------------------------------------------------------------------
# Full model (all three components)
# ---------------------------------------------------------------------------

class TestFullModel:
    def test_full_model_all_flags(self):
        model = (
            ModelBuilder()
            .with_eas(make_eas_config())
            .with_gan(make_gan_config())
            .with_laplace(make_laplace_config())
            .build()
        )
        assert model.has_eas
        assert model.has_gan
        assert model.has_laplace

    def test_full_model_repr(self):
        model = (
            ModelBuilder()
            .with_eas(make_eas_config())
            .with_gan(make_gan_config())
            .with_laplace(make_laplace_config())
            .build()
        )
        r = repr(model)
        assert "EAS" in r
        assert "GAN" in r
        assert "Laplace" in r

    def test_full_model_component_summary(self):
        model = (
            ModelBuilder()
            .with_eas(make_eas_config())
            .with_gan(make_gan_config())
            .with_laplace(make_laplace_config())
            .build()
        )
        summary = model.component_summary()
        assert set(summary["active_components"]) == {"eas", "gan", "laplace"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_with_eas_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            ModelBuilder().with_eas({})

    def test_with_eas_missing_num_classes_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            ModelBuilder().with_eas({"population_size": 10, "num_generations": 5})

    def test_with_gan_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            ModelBuilder().with_gan({})

    def test_with_laplace_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            ModelBuilder().with_laplace({})

    def test_eas_invalid_population_size_raises(self):
        """EASEngine._validate_config raises if population_size < 4."""
        with pytest.raises(ValueError):
            ModelBuilder().with_eas(make_eas_config(population_size=2)).build()


# ---------------------------------------------------------------------------
# Fluent API
# ---------------------------------------------------------------------------

class TestFluentAPI:
    def test_with_eas_returns_builder(self):
        builder = ModelBuilder()
        result = builder.with_eas(make_eas_config())
        assert result is builder

    def test_with_gan_returns_builder(self):
        builder = ModelBuilder()
        result = builder.with_gan(make_gan_config())
        assert result is builder

    def test_with_laplace_returns_builder(self):
        builder = ModelBuilder()
        result = builder.with_laplace(make_laplace_config())
        assert result is builder

    def test_chained_call_returns_builder(self):
        builder = ModelBuilder()
        result = builder.with_eas(make_eas_config()).with_gan(make_gan_config())
        assert result is builder


# ---------------------------------------------------------------------------
# All 7 ablation configurations
# ---------------------------------------------------------------------------

class TestAblationConfigurations:
    """Verify all 7 non-trivial ablation combinations build without error."""

    def test_build_all_ablations(self):
        ablations = ModelBuilder.build_all_ablations(
            eas_config=make_eas_config(),
            gan_config=make_gan_config(),
            laplace_config=make_laplace_config(),
        )
        assert set(ablations.keys()) == {
            "eas_only", "gan_only", "laplace_only",
            "eas_gan", "eas_laplace", "gan_laplace", "full",
        }
        for name, model in ablations.items():
            assert isinstance(model, SulcalSegmentationModel), f"{name} is not a model"

    def test_eas_only_flags(self):
        ablations = ModelBuilder.build_all_ablations(
            eas_config=make_eas_config(),
            gan_config=make_gan_config(),
            laplace_config=make_laplace_config(),
        )
        m = ablations["eas_only"]
        assert m.has_eas and not m.has_gan and not m.has_laplace

    def test_gan_only_flags(self):
        ablations = ModelBuilder.build_all_ablations(
            eas_config=make_eas_config(),
            gan_config=make_gan_config(),
            laplace_config=make_laplace_config(),
        )
        m = ablations["gan_only"]
        assert not m.has_eas and m.has_gan and not m.has_laplace

    def test_laplace_only_flags(self):
        ablations = ModelBuilder.build_all_ablations(
            eas_config=make_eas_config(),
            gan_config=make_gan_config(),
            laplace_config=make_laplace_config(),
        )
        m = ablations["laplace_only"]
        assert not m.has_eas and not m.has_gan and m.has_laplace

    def test_full_flags(self):
        ablations = ModelBuilder.build_all_ablations(
            eas_config=make_eas_config(),
            gan_config=make_gan_config(),
            laplace_config=make_laplace_config(),
        )
        m = ablations["full"]
        assert m.has_eas and m.has_gan and m.has_laplace
