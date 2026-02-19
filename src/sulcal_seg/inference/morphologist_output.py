"""Convert segmentation arrays to Morphologist-compatible output format."""
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


def segmentation_to_morphologist_format(
    segmentation: np.ndarray,
    output_dir: Union[str, Path],
    mri_path: Optional[Union[str, Path]] = None,
    subject_id: str = "unknown",
) -> Dict[str, Path]:
    """
    Convert a segmentation array to Morphologist-compatible output files.

    Writes three artefacts to ``output_dir``:

    * ``<subject_id>.arg``
        HDF5 sulcal graph with ``/metadata/`` and ``/sulci/<label>/``
        groups containing vertices, faces, normals, and attributes.
    * ``<subject_id>_metadata.json``
        Algorithm parameters and per-sulcus statistics (label, volume,
        centroid, surface area).
    * ``<subject_id>_segmentation.nii.gz``
        Argmax label volume in NIfTI format.

    Args:
        segmentation: Probability or logit array of shape
            (num_classes, D, H, W) or (D, H, W) for hard labels.
        output_dir: Directory where output files will be written.
        mri_path: Optional path to the source T1 MRI (stored in metadata).
        subject_id: Identifier used as the filename stem.

    Returns:
        Dict mapping ``'arg'``, ``'metadata'``, ``'segmentation'`` to the
        corresponding output ``Path`` objects.

    Raises:
        NotImplementedError: Always — full implementation requires
            scikit-image (marching cubes), h5py (HDF5 graph writing),
            and nibabel (NIfTI export).
    """
    raise NotImplementedError(
        "segmentation_to_morphologist_format() requires scikit-image, "
        "h5py, and nibabel. Implement the mesh extraction and HDF5 "
        "writing steps before calling this function."
    )
