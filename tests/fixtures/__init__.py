"""Test fixtures and data factories."""
from .sample_data import make_synthetic_image, make_synthetic_label, make_synthetic_nifti_dir

__all__ = [
    "make_synthetic_image",
    "make_synthetic_label",
    "make_synthetic_nifti_dir",
]
