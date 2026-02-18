"""Abstract base class for all dataset loaders."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class AbstractDataLoader(ABC):
    """
    Abstract base class for all dataset implementations.

    Any dataset can be plugged into the training pipeline by subclassing
    this class and implementing the abstract methods. The model code
    never needs to know which dataset is being used.

    Required directory structure (per dataset):
        root_dir/
        ├── train/
        │   ├── images/          (*.nii.gz T1 volumes)
        │   ├── morphologist_labels/  (*.nii.gz segmentation masks)
        │   └── freesurfer_labels/   (*.nii.gz segmentation masks, optional subset)
        ├── val/  (same structure)
        └── test/ (same structure)
    """

    def __init__(self, root_dir: Path, subset: str = "train"):
        """
        Args:
            root_dir: Root directory containing dataset
            subset: One of 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self._validate_subset(subset)

    @abstractmethod
    def _validate_subset(self, subset: str) -> None:
        """Validate that subset is valid and directory exists."""

    @abstractmethod
    def get_subject_ids(self) -> List[str]:
        """
        Return list of all subject IDs in subset.

        Returns:
            Sorted list of unique subject identifiers
        """

    @abstractmethod
    def load_image(self, subject_id: str) -> np.ndarray:
        """
        Load T1-weighted MRI image for subject.

        Args:
            subject_id: Subject identifier

        Returns:
            3D image array [H, W, D], typically ~(256, 256, 256)
        """

    @abstractmethod
    def load_morphologist_label(self, subject_id: str) -> np.ndarray:
        """
        Load Morphologist segmentation label for subject.

        Returns:
            3D integer label array [H, W, D]
        """

    @abstractmethod
    def load_freesurfer_label(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Load FreeSurfer segmentation label for subject.

        Returns:
            3D integer label array [H, W, D], or None if not available.
        """

    @abstractmethod
    def get_metadata(self, subject_id: str) -> Dict:
        """
        Load metadata for subject.

        Returns:
            Dict with at minimum:
            {
                'age': float,
                'sex': str ('M' or 'F'),
                'scanner': str ('Siemens', 'GE', 'Philips'),
                'site': str,
                'group': str ('healthy', 'ASD', etc.)
            }
        """

    def __len__(self) -> int:
        """Return number of subjects in subset."""
        return len(self.get_subject_ids())

    def __getitem__(self, idx: int) -> Dict:
        """
        Get subject data by index.

        Returns:
            {
                'subject_id': str,
                'image': np.ndarray [H, W, D],
                'morphologist_label': np.ndarray [H, W, D],
                'freesurfer_label': Optional[np.ndarray],
                'metadata': Dict
            }
        """
        subject_id = self.get_subject_ids()[idx]
        return {
            "subject_id": subject_id,
            "image": self.load_image(subject_id),
            "morphologist_label": self.load_morphologist_label(subject_id),
            "freesurfer_label": self.load_freesurfer_label(subject_id),
            "metadata": self.get_metadata(subject_id),
        }
