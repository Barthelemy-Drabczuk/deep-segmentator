"""Synthetic data factories for unit tests."""
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np


def make_synthetic_image(
    shape: Tuple[int, int, int] = (16, 16, 16),
    seed: int = 0,
) -> np.ndarray:
    """
    Create a synthetic 3D T1 MRI image (float32).

    Args:
        shape: Spatial dimensions (D, H, W).
        seed: Random seed.

    Returns:
        (D, H, W) float32 array with non-zero variance.
    """
    rng = np.random.RandomState(seed)
    # Gaussian noise + low-frequency spatial signal
    image = rng.randn(*shape).astype(np.float32)
    # Add a bright centre blob to ensure non-trivial std
    cx, cy, cz = [s // 2 for s in shape]
    image[cx - 2 : cx + 2, cy - 2 : cy + 2, cz - 2 : cz + 2] += 5.0
    return image


def make_synthetic_label(
    shape: Tuple[int, int, int] = (16, 16, 16),
    num_classes: int = 4,
    seed: int = 1,
) -> np.ndarray:
    """
    Create a synthetic integer segmentation label map.

    Args:
        shape: Spatial dimensions (D, H, W).
        num_classes: Number of distinct labels (0 = background).
        seed: Random seed.

    Returns:
        (D, H, W) int32 array with values in [0, num_classes-1].
    """
    rng = np.random.RandomState(seed)
    label = rng.randint(0, num_classes, size=shape, dtype=np.int32)
    return label


def make_synthetic_nifti_dir(
    base_dir: Path,
    n_subjects: int = 3,
    shape: Tuple[int, int, int] = (16, 16, 16),
    num_classes: int = 4,
    affine: Optional[np.ndarray] = None,
    subset: str = "train",
) -> Path:
    """
    Create a directory of synthetic NIfTI files matching UKBiobankLoader layout.

    Layout::

        base_dir/
            train/
                images/
                    sub-001.nii.gz
                    sub-002.nii.gz
                    ...
                morphologist_labels/
                    sub-001.nii.gz
                    ...
                freesurfer_labels/
                    sub-001.nii.gz
                    ...

    Args:
        base_dir: Root directory to populate.
        n_subjects: Number of subjects to create.
        shape: Volume shape.
        num_classes: Number of label classes.
        affine: NIfTI affine matrix (defaults to identity).
        subset: Which split to create ('train', 'val', 'test').

    Returns:
        The populated base_dir.
    """
    if affine is None:
        affine = np.eye(4)

    images_dir = base_dir / subset / "images"
    morph_dir = base_dir / subset / "morphologist_labels"
    fs_dir = base_dir / subset / "freesurfer_labels"
    for d in [images_dir, morph_dir, fs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_subjects + 1):
        subject_id = f"sub-{i:03d}"

        # T1 image
        image_data = make_synthetic_image(shape=shape, seed=i)
        nib.save(
            nib.Nifti1Image(image_data, affine),
            str(images_dir / f"{subject_id}.nii.gz"),
        )

        # Morphologist labels
        morph_data = make_synthetic_label(shape=shape, num_classes=num_classes, seed=i + 100)
        nib.save(
            nib.Nifti1Image(morph_data.astype(np.float32), affine),
            str(morph_dir / f"{subject_id}.nii.gz"),
        )

        # FreeSurfer labels
        fs_data = make_synthetic_label(shape=shape, num_classes=num_classes, seed=i + 200)
        nib.save(
            nib.Nifti1Image(fs_data.astype(np.float32), affine),
            str(fs_dir / f"{subject_id}.nii.gz"),
        )

    return base_dir


def make_synthetic_hcp_dir(
    base_dir: Path,
    n_subjects: int = 3,
    shape: Tuple[int, int, int] = (8, 8, 8),
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Create a minimal HCP-like directory tree for testing.

    Layout::

        base_dir/{subject_id}/t1mri/BL/
            {subject_id}.nii.gz
            default_analysis/segmentation/
                skull_stripped_{subject_id}.nii.gz
                Lgrey_white_{subject_id}.nii.gz
                Rgrey_white_{subject_id}.nii.gz
                Lskeleton_{subject_id}.nii.gz
                Rskeleton_{subject_id}.nii.gz

    Subject IDs: '100000', '100001', ...

    Args:
        base_dir:   Root directory to populate (becomes ``root_dir`` for
                    HCPLoader).
        n_subjects: Number of subjects to create.
        shape:      Volume shape for all NIfTI files.
        affine:     NIfTI affine matrix (defaults to identity).

    Returns:
        The populated ``base_dir``.
    """
    if affine is None:
        affine = np.eye(4)

    for i in range(n_subjects):
        sid = f"10{i:04d}"  # "100000", "100001", ...
        bl_dir = base_dir / sid / "t1mri" / "BL"
        seg_dir = bl_dir / "default_analysis" / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)

        img_data = make_synthetic_image(shape=shape, seed=i)
        label_data = make_synthetic_label(
            shape=shape, num_classes=3, seed=i + 100
        )

        # Raw T1
        nib.save(
            nib.Nifti1Image(img_data, affine),
            str(bl_dir / f"{sid}.nii.gz"),
        )
        # Skull-stripped T1
        nib.save(
            nib.Nifti1Image(img_data, affine),
            str(seg_dir / f"skull_stripped_{sid}.nii.gz"),
        )
        # L/R grey-white masks
        nib.save(
            nib.Nifti1Image(label_data.astype(np.float32), affine),
            str(seg_dir / f"Lgrey_white_{sid}.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(label_data.astype(np.float32), affine),
            str(seg_dir / f"Rgrey_white_{sid}.nii.gz"),
        )
        # L/R skeletons
        nib.save(
            nib.Nifti1Image(label_data.astype(np.float32), affine),
            str(seg_dir / f"Lskeleton_{sid}.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(label_data.astype(np.float32), affine),
            str(seg_dir / f"Rskeleton_{sid}.nii.gz"),
        )

    return base_dir
