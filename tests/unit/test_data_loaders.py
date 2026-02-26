"""Unit tests for data loaders."""
import pytest

from sulcal_seg.data.dataset_manager import DatasetLoader
from sulcal_seg.data.loaders.hcp_loader import HCPLoader
from sulcal_seg.data.loaders.ukbiobank_loader import UKBiobankLoader


class TestUKBioBankLoader:
    def test_subject_ids_discovered(self, tmp_ukbiobank_dir):
        loader = UKBiobankLoader(root_dir=str(tmp_ukbiobank_dir), subset="train")
        ids = loader.get_subject_ids()
        assert len(ids) == 3

    def test_subject_id_parsing(self, tmp_ukbiobank_dir):
        """Subject IDs should NOT contain .nii or .nii.gz suffixes."""
        loader = UKBiobankLoader(root_dir=str(tmp_ukbiobank_dir), subset="train")
        ids = loader.get_subject_ids()
        for sid in ids:
            assert ".nii" not in sid, f"Subject ID contains .nii: {sid!r}"
            assert ".gz" not in sid, f"Subject ID contains .gz: {sid!r}"

    def test_load_image_returns_3d_array(self, tmp_ukbiobank_dir):
        import numpy as np

        loader = UKBiobankLoader(root_dir=str(tmp_ukbiobank_dir), subset="train")
        sid = loader.get_subject_ids()[0]
        image = loader.load_image(sid)
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3

    def test_len_matches_subjects(self, tmp_ukbiobank_dir):
        loader = UKBiobankLoader(root_dir=str(tmp_ukbiobank_dir), subset="train")
        assert len(loader) == 3

    def test_getitem_returns_dict_with_required_keys(self, tmp_ukbiobank_dir):
        loader = UKBiobankLoader(root_dir=str(tmp_ukbiobank_dir), subset="train")
        item = loader[0]
        assert "subject_id" in item
        assert "image" in item
        assert "morphologist_label" in item
        assert "freesurfer_label" in item
        assert "metadata" in item


class TestHCPLoader:
    def test_subject_ids_discovered(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        assert len(loader.get_subject_ids()) > 0

    def test_subject_ids_are_digit_strings(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        for sid in loader.get_subject_ids():
            assert sid.isdigit()
            assert ".nii" not in sid

    def test_load_image_returns_3d_array(self, tmp_hcp_dir):
        import numpy as np

        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        img = loader.load_image(loader.get_subject_ids()[0])
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_load_morphologist_label_shape_matches_image(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        sid = loader.get_subject_ids()[0]
        assert loader.load_image(sid).shape == (
            loader.load_morphologist_label(sid).shape
        )

    def test_load_freesurfer_label_returns_none(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        sid = loader.get_subject_ids()[0]
        assert loader.load_freesurfer_label(sid) is None

    def test_getitem_has_required_keys(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        item = loader[0]
        required = [
            "subject_id", "image", "morphologist_label",
            "freesurfer_label", "metadata",
        ]
        assert all(k in item for k in required)

    def test_metadata_defaults_present(self, tmp_hcp_dir):
        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        meta = loader.get_metadata(loader.get_subject_ids()[0])
        assert all(
            k in meta for k in ["age", "sex", "scanner", "site", "group"]
        )

    def test_split_csv_overrides_auto_split(self, tmp_hcp_dir, tmp_path):
        csv_path = tmp_path / "split.csv"
        first_id = sorted(
            p.name for p in tmp_hcp_dir.iterdir() if p.is_dir()
        )[0]
        csv_path.write_text(f"{first_id}\n")
        loader = HCPLoader(
            root_dir=tmp_hcp_dir, subset="train", split_csv=csv_path
        )
        assert loader.get_subject_ids() == [first_id]

    def test_registered_in_dataset_loader(self, tmp_hcp_dir):
        loader = DatasetLoader.get_loader(
            "hcp", root_dir=tmp_hcp_dir, subset="train"
        )
        assert len(loader) > 0

    def test_load_skeleton_returns_3d_array(self, tmp_hcp_dir):
        import numpy as np

        loader = HCPLoader(root_dir=tmp_hcp_dir, subset="train")
        sid = loader.get_subject_ids()[0]
        skel = loader.load_skeleton(sid, side="L")
        assert isinstance(skel, np.ndarray)
        assert skel.ndim == 3
