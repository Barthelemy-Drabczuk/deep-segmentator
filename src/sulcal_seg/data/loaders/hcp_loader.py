"""HCP (Human Connectome Project) dataset loader."""
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from ..abstract_loader import AbstractDataLoader


class HCPLoader(AbstractDataLoader):
    """
    Loader for HCP (Human Connectome Project) BrainVISA/Morphologist data.

    ``root_dir`` must point to the per-subject directory, e.g.::

        /neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp

    Subjects are discovered as 6-digit numeric subdirectories.  Because HCP
    ships without pre-defined train/val/test splits, a deterministic 70/15/15
    split is applied unless the caller supplies a ``split_csv``.

    Expected per-subject layout::

        root_dir/{subject_id}/t1mri/BL/
            {subject_id}.nii.gz                          # raw T1
            default_analysis/segmentation/
                skull_stripped_{subject_id}.nii.gz        # skull-stripped T1
                Lgrey_white_{subject_id}.nii.gz           # L GM/WM boundary
                Rgrey_white_{subject_id}.nii.gz           # R GM/WM boundary
                Lskeleton_{subject_id}.nii.gz             # L sulcal skeleton
                Rskeleton_{subject_id}.nii.gz             # R sulcal skeleton

    ``participants.csv`` is expected two levels above ``root_dir`` with
    columns ``Subject``, ``Gender``, ``Age``.
    """

    VALID_SUBSETS = ("train", "val", "test")
    _SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
    _SPLIT_SEED = 42

    def __init__(
        self,
        root_dir: Path,
        subset: str = "train",
        split_csv: Optional[Path] = None,
    ) -> None:
        """
        Args:
            root_dir:  Directory whose immediate subdirs are subject folders.
            subset:    One of 'train', 'val', 'test'.
            split_csv: Optional path to a single-column CSV (no header, or
                       first column used) listing subject IDs for *this*
                       subset.  When provided, the automatic 70/15/15 split
                       is bypassed.
        """
        self._split_csv = Path(split_csv) if split_csv is not None else None
        super().__init__(root_dir, subset)
        self._subject_ids: List[str] = self._load_subject_ids()
        self._metadata_df = self._load_metadata()

    # ------------------------------------------------------------------
    # AbstractDataLoader interface
    # ------------------------------------------------------------------

    def _validate_subset(self, subset: str) -> None:
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f"subset must be one of {self.VALID_SUBSETS}, got {subset!r}"
            )
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

    def _load_subject_ids(self) -> List[str]:
        if self._split_csv is not None:
            return self._ids_from_csv(self._split_csv)
        return self._ids_from_auto_split()

    def get_subject_ids(self) -> List[str]:
        return self._subject_ids

    def load_image(self, subject_id: str) -> np.ndarray:
        """Load skull-stripped T1 MRI (falls back to raw T1 if missing)."""
        skull = self._seg_dir(subject_id) / f"skull_stripped_{subject_id}.nii.gz"
        if skull.exists():
            return nib.load(str(skull)).get_fdata()
        raw = self._bl_dir(subject_id) / f"{subject_id}.nii.gz"
        return nib.load(str(raw)).get_fdata()

    def load_morphologist_label(self, subject_id: str) -> np.ndarray:
        """
        Return combined L/R grey-white segmentation.

        Values: 0 = background, 1 = left hemisphere, 2 = right hemisphere.
        """
        seg_dir = self._seg_dir(subject_id)
        l_vol = nib.load(
            str(seg_dir / f"Lgrey_white_{subject_id}.nii.gz")
        ).get_fdata()
        r_vol = nib.load(
            str(seg_dir / f"Rgrey_white_{subject_id}.nii.gz")
        ).get_fdata()
        combined = np.zeros(l_vol.shape, dtype=np.int32)
        combined[l_vol != 0] = 1
        combined[r_vol != 0] = 2
        return combined

    def load_freesurfer_label(self, subject_id: str) -> Optional[np.ndarray]:
        """Returns None — HCP BrainVISA data has no FreeSurfer labels here."""
        return None

    def get_metadata(self, subject_id: str) -> Dict:
        defaults: Dict = {
            "age": 28.0,
            "sex": "M",
            "scanner": "Siemens Connectome Skyra 3T",
            "site": "HCP",
            "group": "healthy",
        }
        if self._metadata_df is not None:
            sid = str(subject_id)
            if sid in self._metadata_df.index:
                row = self._metadata_df.loc[sid]
                defaults["sex"] = str(row.get("Gender", defaults["sex"]))
                try:
                    defaults["age"] = float(row.get("Age", defaults["age"]))
                except (TypeError, ValueError):
                    pass
        return defaults

    # ------------------------------------------------------------------
    # Extra convenience method (not in AbstractDataLoader)
    # ------------------------------------------------------------------

    def load_skeleton(self, subject_id: str, side: str = "L") -> np.ndarray:
        """
        Load the Morphologist sulcal skeleton for one hemisphere.

        Args:
            subject_id: Subject identifier.
            side:       'L' (left) or 'R' (right).
        """
        if side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        path = self._seg_dir(subject_id) / f"{side}skeleton_{subject_id}.nii.gz"
        return nib.load(str(path)).get_fdata()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bl_dir(self, subject_id: str) -> Path:
        return self.root_dir / subject_id / "t1mri" / "BL"

    def _seg_dir(self, subject_id: str) -> Path:
        return self._bl_dir(subject_id) / "default_analysis" / "segmentation"

    def _ids_from_csv(self, csv_path: Path) -> List[str]:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, header=None, dtype=str)
            return sorted(df.iloc[:, 0].str.strip().tolist())
        except Exception:
            with open(csv_path) as fh:
                lines = [ln.strip().split(",")[0] for ln in fh if ln.strip()]
            return sorted(lines)

    def _ids_from_auto_split(self) -> List[str]:
        all_ids = sorted(
            p.name
            for p in self.root_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        n = len(all_ids)
        rng = np.random.default_rng(self._SPLIT_SEED)
        perm = rng.permutation(n)
        shuffled = [all_ids[i] for i in perm]

        n_train = int(n * self._SPLIT_RATIOS["train"])
        n_val = int(n * self._SPLIT_RATIOS["val"])

        if self.subset == "train":
            return sorted(shuffled[:n_train])
        if self.subset == "val":
            return sorted(shuffled[n_train : n_train + n_val])
        return sorted(shuffled[n_train + n_val :])

    def _load_metadata(self):
        csv_path = self.root_dir.parent.parent / "participants.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, index_col="Subject", dtype=str)
                df.index = df.index.astype(str)
                return df
            except Exception:
                pass
        return None
