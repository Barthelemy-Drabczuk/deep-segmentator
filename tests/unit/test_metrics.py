"""Unit tests for validation metrics (pure numpy, no GPU)."""
import numpy as np
import pytest

from sulcal_seg.validation.metrics import (
    compute_all_metrics,
    dice_balance,
    dice_per_label,
    dice_score,
    macro_dice,
    mesh_quality,
    output_entropy,
    sulcal_depth_accuracy,
    sulcal_width_accuracy,
)


class TestDiceScore:
    def test_perfect_dice(self):
        pred = np.ones((4, 4, 4), dtype=bool)
        target = np.ones((4, 4, 4), dtype=bool)
        assert dice_score(pred, target) == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap(self):
        pred = np.zeros((4, 4, 4), dtype=bool)
        pred[:2] = True
        target = np.zeros((4, 4, 4), dtype=bool)
        target[2:] = True
        d = dice_score(pred, target)
        # smooth=1 so not exactly 0
        assert d < 0.1

    def test_empty_pred_and_target(self):
        pred = np.zeros((4, 4, 4), dtype=bool)
        target = np.zeros((4, 4, 4), dtype=bool)
        d = dice_score(pred, target, smooth=1.0)
        assert d == pytest.approx(1.0)  # 2*0+1 / 0+0+1 = 1


class TestDicePerLabel:
    def test_shape(self):
        pred = np.zeros((8, 8, 8), dtype=np.int32)
        target = np.zeros((8, 8, 8), dtype=np.int32)
        scores = dice_per_label(pred, target, num_classes=4)
        assert scores.shape == (4,)

    def test_perfect_per_label(self):
        pred = np.zeros((4, 4, 4), dtype=np.int32)
        target = np.zeros((4, 4, 4), dtype=np.int32)
        scores = dice_per_label(pred, target, num_classes=3)
        assert scores[0] == pytest.approx(1.0, abs=1e-4)


class TestMacroDice:
    def test_perfect_macro_dice(self):
        pred = np.zeros((4, 4, 4), dtype=np.int32)
        target = np.zeros((4, 4, 4), dtype=np.int32)
        # All background
        d = macro_dice(pred, target, num_classes=3, ignore_background=True)
        # Classes 1 and 2 both have 0/0 → smooth Dice ≈ 1
        assert 0.0 <= d <= 1.0


class TestOutputEntropy:
    def test_uniform_dist_high_entropy(self):
        """Uniform probability distribution should yield high entropy."""
        C = 4
        probs = np.full((C, 4, 4, 4), 1.0 / C, dtype=np.float32)
        h = output_entropy(probs)
        expected = np.log(C)  # max entropy
        assert h == pytest.approx(expected, rel=0.01)

    def test_deterministic_dist_low_entropy(self):
        """One-hot probability distribution should yield near-zero entropy."""
        C = 4
        probs = np.zeros((C, 4, 4, 4), dtype=np.float32)
        probs[0] = 1.0  # class 0 always
        h = output_entropy(probs)
        assert h < 0.01


class TestDiceBalance:
    def test_balanced_gives_high_score(self):
        """Equal per-class Dice → low variance → high balance."""
        pred = np.array([0, 1, 2, 3] * (4 * 4), dtype=np.int32).reshape(4, 4, 4)
        target = pred.copy()
        score = dice_balance(pred, target, num_classes=4)
        assert score == pytest.approx(1.0, abs=0.1)


class TestSulcalGeometryMetrics:
    def test_perfect_depth_accuracy(self):
        arr = np.ones((4, 4, 4), dtype=np.float32)
        mae = sulcal_depth_accuracy(arr, arr)
        assert mae == pytest.approx(0.0)

    def test_depth_accuracy_with_offset(self):
        pred = np.ones((4, 4, 4), dtype=np.float32)
        target = pred + 1.0
        mae = sulcal_depth_accuracy(pred, target)
        assert mae == pytest.approx(1.0)

    def test_width_accuracy_with_mask(self):
        pred = np.zeros((4, 4, 4), dtype=np.float32)
        target = np.ones((4, 4, 4), dtype=np.float32)
        mask = np.ones((4, 4, 4), dtype=bool)
        mask[:2] = False  # mask out half
        mae = sulcal_width_accuracy(pred, target, mask=mask)
        assert mae == pytest.approx(1.0)


class TestMeshQuality:
    def test_regular_triangle(self):
        """Equilateral triangle mesh should have positive area and aspect ratio."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        result = mesh_quality(vertices, faces)
        assert result["min_area"] > 0.0
        assert result["mean_area"] > 0.0
        assert result["aspect_ratio_mean"] > 0.0

    def test_empty_faces_returns_zeros(self):
        vertices = np.zeros((3, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)
        result = mesh_quality(vertices, faces)
        assert result["min_area"] == 0.0


class TestComputeAllMetrics:
    def test_returns_dict_with_macro_dice(self):
        pred = np.zeros((8, 8, 8), dtype=np.int32)
        target = np.zeros((8, 8, 8), dtype=np.int32)
        metrics = compute_all_metrics(pred, target, num_classes=4)
        assert "macro_dice" in metrics
        assert "dice_balance" in metrics
        assert 0.0 <= metrics["macro_dice"] <= 1.0

    def test_with_prob_map_adds_entropy(self):
        pred = np.zeros((8, 8, 8), dtype=np.int32)
        target = np.zeros((8, 8, 8), dtype=np.int32)
        probs = np.ones((4, 8, 8, 8), dtype=np.float32) / 4
        metrics = compute_all_metrics(pred, target, num_classes=4, pred_probs=probs)
        assert "output_entropy" in metrics
