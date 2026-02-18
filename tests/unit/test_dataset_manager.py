"""Unit tests for DatasetManager singleton."""
import pytest

from sulcal_seg.data.dataset_manager import DatasetLoader, DatasetManager


class TestDatasetManagerSingleton:
    def test_same_instance(self):
        m1 = DatasetManager()
        m2 = DatasetManager()
        assert m1 is m2

    def test_reset_clears_instance(self):
        m1 = DatasetManager()
        DatasetManager.reset()
        m2 = DatasetManager()
        assert m1 is not m2

    def test_reset_called_by_fixture(self):
        """autouse fixture resets singleton, so each test gets a fresh instance."""
        m = DatasetManager()
        assert m is DatasetManager()

    def test_get_subject_unregistered_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            DatasetManager().get_subject("nonexistent_dataset", "sub-001")


class TestDatasetLoader:
    def test_get_loader_ukbiobank(self, tmp_ukbiobank_dir):
        loader = DatasetLoader.get_loader(
            "ukbiobank",
            root_dir=str(tmp_ukbiobank_dir),
            subset="train",
        )
        assert loader is not None
        assert len(loader) > 0

    def test_get_loader_unknown_raises(self):
        with pytest.raises(ValueError):
            DatasetLoader.get_loader("imaginary_dataset", root_dir="/tmp")

    def test_register_custom_loader(self, tmp_ukbiobank_dir):
        from sulcal_seg.data.loaders.ukbiobank_loader import UKBiobankLoader

        DatasetLoader.register("test_custom_abc", UKBiobankLoader)
        loader = DatasetLoader.get_loader(
            "test_custom_abc",
            root_dir=str(tmp_ukbiobank_dir),
            subset="train",
        )
        assert len(loader) == 3
        # Cleanup: remove to avoid polluting other tests
        del DatasetLoader.LOADERS["test_custom_abc"]
