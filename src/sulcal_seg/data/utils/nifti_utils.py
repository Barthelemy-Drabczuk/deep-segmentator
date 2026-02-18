"""NIfTI file handling utilities using nibabel."""
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return its data and affine matrix.

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        (data, affine) where data is np.ndarray and affine is 4x4 matrix
    """
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """
    Save a numpy array as a NIfTI file.

    Args:
        data: Array to save
        affine: 4x4 affine transformation matrix
        path: Output path (will create parent directories)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def get_voxel_size(path: Path) -> Tuple[float, ...]:
    """
    Get the voxel size (resolution) of a NIfTI file in mm.

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        Tuple of (x_mm, y_mm, z_mm) voxel dimensions
    """
    img = nib.load(str(path))
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def get_image_shape(path: Path) -> Tuple[int, ...]:
    """Get the shape of a NIfTI image without loading full data."""
    img = nib.load(str(path))
    return tuple(img.shape)


def is_nifti_valid(path: Path) -> bool:
    """Check if a NIfTI file can be loaded without errors."""
    try:
        img = nib.load(str(path))
        _ = img.header
        return True
    except Exception:
        return False
