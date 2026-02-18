"""
Custom dataset loader template.

Copy this file and implement all abstract methods to add a new dataset.
Register your loader with:
    DatasetLoader.register('my_dataset', CustomLoader)
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..abstract_loader import AbstractDataLoader


class CustomLoader(AbstractDataLoader):
    """
    Template for custom dataset loaders.

    To add a new dataset:
    1. Copy this file to e.g. `my_dataset_loader.py`
    2. Rename the class (e.g. `MyDatasetLoader`)
    3. Set VALID_SUBSETS for your dataset
    4. Implement all TODO methods below
    5. Register: DatasetLoader.register('my_dataset', MyDatasetLoader)
    6. Create configs/datasets/my_dataset.yaml

    Required directory structure:
        root_dir/
        ├── train/
        │   ├── images/               (*.nii.gz T1 volumes)
        │   ├── morphologist_labels/  (*.nii.gz segmentation masks)
        │   └── freesurfer_labels/    (optional)
        ├── val/
        └── test/
    """

    VALID_SUBSETS = ("train", "val", "test")

    def __init__(self, root_dir: Path, subset: str = "train"):
        super().__init__(root_dir, subset)
        self._subject_ids: List[str] = self._load_subject_ids()

    def _validate_subset(self, subset: str) -> None:
        if subset not in self.VALID_SUBSETS:
            raise ValueError(f"subset must be one of {self.VALID_SUBSETS}, got {subset!r}")
        subset_dir = self.root_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    def _load_subject_ids(self) -> List[str]:
        # TODO: Customize subject ID discovery if your filenames differ
        image_dir = self.root_dir / self.subset / "images"
        return sorted(
            p.name.replace(".nii.gz", "").replace(".nii", "")
            for p in image_dir.glob("*.nii.gz")
        )

    def get_subject_ids(self) -> List[str]:
        return self._subject_ids

    def load_image(self, subject_id: str) -> np.ndarray:
        # TODO: Customize if your image paths differ
        import nibabel as nib
        path = self.root_dir / self.subset / "images" / f"{subject_id}.nii.gz"
        return nib.load(str(path)).get_fdata()

    def load_morphologist_label(self, subject_id: str) -> np.ndarray:
        # TODO: Customize if your label paths differ
        import nibabel as nib
        path = self.root_dir / self.subset / "morphologist_labels" / f"{subject_id}.nii.gz"
        return nib.load(str(path)).get_fdata()

    def load_freesurfer_label(self, subject_id: str) -> Optional[np.ndarray]:
        # TODO: Return None if FreeSurfer labels are not available
        import nibabel as nib
        path = self.root_dir / self.subset / "freesurfer_labels" / f"{subject_id}.nii.gz"
        if path.exists():
            return nib.load(str(path)).get_fdata()
        return None

    def get_metadata(self, subject_id: str) -> Dict:
        # TODO: Load from your metadata CSV or return defaults
        return {
            "age": 0.0,      # TODO: fill in
            "sex": "M",      # TODO: fill in
            "scanner": "Unknown",
            "site": "CustomSite",
            "group": "healthy",
        }
