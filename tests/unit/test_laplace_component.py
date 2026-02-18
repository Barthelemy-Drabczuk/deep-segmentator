"""Unit tests for Laplace component (no GPU, topology validator fully testable)."""
import pytest
import numpy as np
import pydantic

# Config classes require Pydantic v2 features
PYDANTIC_V2 = int(pydantic.VERSION.split(".")[0]) >= 2


class TestLaplaceComponentConfig:
    @pytest.mark.skipif(not PYDANTIC_V2, reason="requires pydantic>=2")
    def test_valid_config(self):
        from sulcal_seg.models.components.laplace.config import LaplaceComponentConfig
        cfg = LaplaceComponentConfig(num_iterations=30, lambda_smooth=0.5)
        assert cfg.num_iterations == 30
        assert cfg.lambda_smooth == 0.5

    @pytest.mark.skipif(not PYDANTIC_V2, reason="requires pydantic>=2")
    def test_invalid_lambda_raises(self):
        from sulcal_seg.models.components.laplace.config import LaplaceComponentConfig
        with pytest.raises(Exception):
            LaplaceComponentConfig(num_iterations=10, lambda_smooth=0.0)

    @pytest.mark.skipif(not PYDANTIC_V2, reason="requires pydantic>=2")
    def test_invalid_lambda_above_1_raises(self):
        from sulcal_seg.models.components.laplace.config import LaplaceComponentConfig
        with pytest.raises(Exception):
            LaplaceComponentConfig(num_iterations=10, lambda_smooth=1.5)

    @pytest.mark.skipif(not PYDANTIC_V2, reason="requires pydantic>=2")
    def test_default_algorithm(self):
        from sulcal_seg.models.components.laplace.config import LaplaceComponentConfig
        cfg = LaplaceComponentConfig(num_iterations=10, lambda_smooth=0.3)
        assert cfg.algorithm == "taubin"


class TestTopologyValidator:
    def _make_tetrahedron(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=np.int32)
        return vertices, faces

    def test_valence_histogram_shape(self):
        from sulcal_seg.models.components.laplace.validators import TopologyValidator
        validator = TopologyValidator()
        vertices, faces = self._make_tetrahedron()
        valence = validator.compute_valence_histogram(faces, len(vertices))
        assert valence.shape == (4,)

    def test_valence_tetrahedron(self):
        from sulcal_seg.models.components.laplace.validators import TopologyValidator
        validator = TopologyValidator()
        vertices, faces = self._make_tetrahedron()
        valence = validator.compute_valence_histogram(faces, len(vertices))
        assert np.all(valence == 3)

    def test_euler_characteristic_tetrahedron(self):
        from sulcal_seg.models.components.laplace.validators import TopologyValidator
        validator = TopologyValidator(target_genus=0)
        vertices, faces = self._make_tetrahedron()
        result = validator.check_euler_characteristic(vertices, faces)
        assert result["V"] == 4
        assert result["E"] == 6
        assert result["F"] == 4
        assert result["euler"] == 2
        assert result["genus"] == 0
        assert result["is_target_genus"] is True

    def test_validate_returns_dict(self):
        from sulcal_seg.models.components.laplace.validators import TopologyValidator
        validator = TopologyValidator(target_genus=0)
        vertices, faces = self._make_tetrahedron()
        result = validator.validate(vertices, faces)
        assert "euler" in result
        assert "valence_mean" in result
        assert "passed" in result
        assert result["passed"] is True

    def test_holes_check_raises_not_implemented(self):
        from sulcal_seg.models.components.laplace.validators import TopologyValidator
        validator = TopologyValidator()
        vertices, faces = self._make_tetrahedron()
        with pytest.raises(NotImplementedError):
            validator.check_holes(vertices, faces)
