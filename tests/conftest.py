"""Pytest configuration and shared fixtures."""
import pytest

from sulcal_seg.data.dataset_manager import DatasetManager


@pytest.fixture(autouse=True)
def reset_dataset_manager():
    """Reset the DatasetManager singleton before and after each test."""
    DatasetManager.reset()
    yield
    DatasetManager.reset()


@pytest.fixture
def tmp_hcp_dir(tmp_path):
    """Synthetic HCP-like directory with 3 subjects."""
    from tests.fixtures.sample_data import make_synthetic_hcp_dir

    return make_synthetic_hcp_dir(
        base_dir=tmp_path / "hcp",
        n_subjects=3,
        shape=(8, 8, 8),
    )


@pytest.fixture
def tmp_ukbiobank_dir(tmp_path):
    """Synthetic UKBioBank-like directory with 3 subjects and NIfTI files."""
    from tests.fixtures.sample_data import make_synthetic_nifti_dir

    return make_synthetic_nifti_dir(
        base_dir=tmp_path / "ukbiobank",
        n_subjects=3,
        shape=(8, 8, 8),
        num_classes=4,
    )
