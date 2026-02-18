"""Validation metrics for sulcal segmentation.

All metrics are implemented in pure NumPy/SciPy — no GPU required.
"""
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.spatial.distance import directed_hausdorff
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Binary Dice score for a single class.

    Args:
        pred: Binary prediction array (0/1), any shape.
        target: Binary ground-truth array, same shape.
        smooth: Laplace smoothing term.

    Returns:
        Dice score in [0, 1].
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def dice_per_label(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
    smooth: float = 1.0,
) -> np.ndarray:
    """
    Compute per-class Dice for a multi-class integer segmentation.

    Args:
        pred: Integer label map (D, H, W).
        target: Integer ground-truth map (D, H, W).
        num_classes: Total number of classes.
        ignore_index: Class to exclude (-1 = include all).
        smooth: Laplace smoothing term.

    Returns:
        (num_classes,) array of Dice scores.
    """
    scores = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if c == ignore_index:
            continue
        p = (pred == c).ravel()
        t = (target == c).ravel()
        intersection = np.logical_and(p, t).sum()
        scores[c] = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return scores


def macro_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> float:
    """
    Macro-average Dice: mean of per-class Dice scores.

    Args:
        pred: Integer label map.
        target: Integer ground-truth map.
        num_classes: Total number of classes.
        ignore_background: If True, class 0 (background) is excluded.

    Returns:
        Macro-average Dice in [0, 1].
    """
    ignore = 0 if ignore_background else -1
    per_class = dice_per_label(pred, target, num_classes, ignore_index=ignore)
    start = 1 if ignore_background else 0
    return float(per_class[start:].mean())


def hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Hausdorff distance between binary surface voxels.

    Args:
        pred: Binary prediction array (D, H, W).
        target: Binary ground-truth array (D, H, W).
        percentile: Use percentile Hausdorff (HD95 by default) instead of max.

    Returns:
        Hausdorff distance (lower = better).

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If either surface is empty.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for hausdorff_distance()")

    pred_pts = np.argwhere(pred.astype(bool))
    target_pts = np.argwhere(target.astype(bool))

    if len(pred_pts) == 0 or len(target_pts) == 0:
        raise ValueError("One of the surfaces is empty — cannot compute Hausdorff distance.")

    d_forward = directed_hausdorff(pred_pts, target_pts)[0]
    d_backward = directed_hausdorff(target_pts, pred_pts)[0]

    if percentile == 100.0:
        return float(max(d_forward, d_backward))

    # HD95: use per-point minimum distances instead of max
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree_target = cKDTree(target_pts)
    tree_pred = cKDTree(pred_pts)

    dist_fwd, _ = tree_target.query(pred_pts, k=1)
    dist_bwd, _ = tree_pred.query(target_pts, k=1)
    all_dists = np.concatenate([dist_fwd, dist_bwd])
    return float(np.percentile(all_dists, percentile))


# ---------------------------------------------------------------------------
# Sulcal geometry metrics
# ---------------------------------------------------------------------------

def sulcal_depth_accuracy(
    pred_depth: np.ndarray,
    target_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean absolute error between predicted and reference sulcal depth maps.

    Args:
        pred_depth: Predicted sulcal depth map (D, H, W) in mm.
        target_depth: Reference depth map (D, H, W) in mm.
        mask: Optional binary mask to restrict evaluation region.

    Returns:
        Mean absolute error (lower = better).
    """
    if mask is not None:
        pred_depth = pred_depth[mask.astype(bool)]
        target_depth = target_depth[mask.astype(bool)]
    return float(np.abs(pred_depth - target_depth).mean())


def sulcal_width_accuracy(
    pred_width: np.ndarray,
    target_width: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean absolute error between predicted and reference sulcal width maps.

    Args:
        pred_width: Predicted sulcal width map (D, H, W) in mm.
        target_width: Reference width map (D, H, W) in mm.
        mask: Optional binary mask.

    Returns:
        Mean absolute error (lower = better).
    """
    if mask is not None:
        pred_width = pred_width[mask.astype(bool)]
        target_width = target_width[mask.astype(bool)]
    return float(np.abs(pred_width - target_width).mean())


# ---------------------------------------------------------------------------
# Mode collapse metrics
# ---------------------------------------------------------------------------

def output_entropy(probs: np.ndarray) -> float:
    """
    Mean Shannon entropy of a batch of segmentation probability maps.

    Args:
        probs: Softmax probabilities (B, C, D, H, W) or (C, D, H, W).

    Returns:
        Mean entropy (higher = more diverse output).
    """
    eps = 1e-8
    entropy = -(probs * np.log(probs + eps)).sum(axis=0 if probs.ndim == 4 else 1)
    return float(entropy.mean())


def ndb_score(
    real_samples: np.ndarray,
    fake_samples: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Number of Statistically Different Bins (NDB) score.

    Partitions the real data into Voronoi bins and counts how many bins
    have a significantly different proportion of fake samples.

    Args:
        real_samples: (N, D) real feature vectors.
        fake_samples: (M, D) generated feature vectors.
        n_bins: Number of k-means clusters used as bins.

    Returns:
        NDB score (fraction of significantly different bins, lower = better).

    Raises:
        NotImplementedError: Always (stub — requires sklearn KMeans + scipy chi²).
    """
    # TODO: Implement NDB:
    #   1. Cluster real_samples with KMeans(n_bins)
    #   2. Assign real and fake to nearest centroid
    #   3. For each bin, chi² test: real count vs fake count
    #   4. Count bins where H0 rejected at alpha=0.05
    #   5. Return count / n_bins
    raise NotImplementedError(
        "ndb_score() requires sklearn.cluster.KMeans and scipy.stats.chi2_contingency. "
        "Implement NDB computation here."
    )


def dice_balance(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> float:
    """
    Measure mode-collapse in the class distribution by comparing
    the coefficient of variation (CV) of per-class Dice scores.

    Low Dice balance → high variance → some classes are collapsed.

    Args:
        pred: Integer label map.
        target: Integer ground-truth map.
        num_classes: Number of classes.

    Returns:
        1 - CV (higher = more balanced, less mode collapse).
    """
    per_class = dice_per_label(pred, target, num_classes, ignore_index=0)[1:]
    if per_class.std() == 0.0:
        return 1.0
    cv = per_class.std() / (per_class.mean() + 1e-8)
    return float(1.0 - min(cv, 1.0))


# ---------------------------------------------------------------------------
# Mesh quality
# ---------------------------------------------------------------------------

def mesh_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, float]:
    """
    Compute basic triangle mesh quality metrics.

    Args:
        vertices: (N, 3) vertex positions.
        faces: (M, 3) face indices.

    Returns:
        Dict with keys: 'min_area', 'mean_area', 'aspect_ratio_mean'.
    """
    areas = []
    aspect_ratios = []

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)
        # Area via cross product
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross)
        areas.append(area)
        # Aspect ratio: longest edge / shortest altitude ≈ max_edge / (2*area / max_edge)
        max_edge = max(e0, e1, e2)
        aspect = max_edge ** 2 / (2.0 * area + 1e-8)
        aspect_ratios.append(aspect)

    return {
        "min_area": float(np.min(areas)) if areas else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
    }


def inference_latency(
    model: object,
    sample_input: object,
    n_runs: int = 10,
) -> Dict[str, float]:
    """
    Measure model inference latency.

    Args:
        model: PyTorch model with a .forward() method.
        sample_input: Sample input tensor.
        n_runs: Number of timed forward passes.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms'.
    """
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.forward(sample_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
    }


# ---------------------------------------------------------------------------
# Aggregated metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 41,
    pred_probs: Optional[np.ndarray] = None,
    ignore_background: bool = True,
) -> Dict[str, float]:
    """
    Compute all available deterministic metrics.

    Args:
        pred: Integer label map (D, H, W).
        target: Integer ground-truth map (D, H, W).
        num_classes: Number of segmentation classes.
        pred_probs: Optional probability map (C, D, H, W) for entropy.
        ignore_background: Exclude background from macro Dice.

    Returns:
        Dict mapping metric name to value.
    """
    results: Dict[str, float] = {}

    results["macro_dice"] = macro_dice(pred, target, num_classes, ignore_background)
    results["dice_balance"] = dice_balance(pred, target, num_classes)

    if HAS_SCIPY:
        try:
            # Background-subtracted surface for HD
            pred_fg = (pred > 0).astype(np.uint8)
            target_fg = (target > 0).astype(np.uint8)
            results["hausdorff_95"] = hausdorff_distance(pred_fg, target_fg, percentile=95.0)
        except ValueError:
            results["hausdorff_95"] = float("nan")

    if pred_probs is not None:
        results["output_entropy"] = output_entropy(pred_probs)

    return results
