"""ABIDE (Autism Brain Imaging Data Exchange) dataset loader."""
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from ..abstract_loader import AbstractDataLoader


class ABIDELoader(AbstractDataLoader):
    """
    Data loader for ABIDE cohort (clinical ASD population).

    Expected directory structure: same as UKBiobankLoader.
    Metadata CSV additionally expected to have: clinical_diagnosis column.

    Stats:
        - 1,100 subjects (539 ASD, 573 controls)
        - Age range: 7-64 years
        - Multiple international sites, multiple scanner vendors
        - Labels: Morphologist + FreeSurfer available
    """

    VALID_SUBSETS = ("train", "val", "test")
    VALID_DIAGNOSES = ("ASD", "control")

    def __init__(self, root_dir: Path, subset: str = "test"):
        super().__init__(root_dir, subset)
        self._subject_ids: List[str] = self._load_subject_ids()
        self._metadata_df = self._load_metadata()

    def _validate_subset(self, subset: str) -> None:
        if subset not in self.VALID_SUBSETS:
            raise ValueError(f"subset must be one of {self.VALID_SUBSETS}, got {subset!r}")
        subset_dir = self.root_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    def _load_subject_ids(self) -> List[str]:
        image_dir = self.root_dir / self.subset / "images"
        return sorted(
            p.name.replace(".nii.gz", "").replace(".nii", "")
            for p in image_dir.glob("*.nii.gz")
        )

    def _load_metadata(self):
        csv_path = self.root_dir / "metadata.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                return pd.read_csv(csv_path, index_col="subject_id")
            except Exception:
                pass
        return None

    def get_subject_ids(self) -> List[str]:
        return self._subject_ids

    def load_image(self, subject_id: str) -> np.ndarray:
        path = self.root_dir / self.subset / "images" / f"{subject_id}.nii.gz"
        img = nib.load(str(path))
        return img.get_fdata()

    def load_morphologist_label(self, subject_id: str) -> np.ndarray:
        path = self.root_dir / self.subset / "morphologist_labels" / f"{subject_id}.nii.gz"
        img = nib.load(str(path))
        return img.get_fdata()

    def load_freesurfer_label(self, subject_id: str) -> Optional[np.ndarray]:
        path = self.root_dir / self.subset / "freesurfer_labels" / f"{subject_id}.nii.gz"
        if path.exists():
            img = nib.load(str(path))
            return img.get_fdata()
        return None

    def get_metadata(self, subject_id: str) -> Dict:
        defaults = {
            "age": 20.0,
            "sex": "M",
            "scanner": "Siemens",
            "site": "ABIDE_site_01",
            "group": "control",
            "clinical_diagnosis": "control",
        }
        if self._metadata_df is not None and subject_id in self._metadata_df.index:
            row = self._metadata_df.loc[subject_id]
            return {k: row.get(k, defaults[k]) for k in defaults}
        return defaults
