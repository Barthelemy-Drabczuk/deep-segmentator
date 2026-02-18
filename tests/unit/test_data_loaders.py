"""Unit tests for data loaders."""
import pytest

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
