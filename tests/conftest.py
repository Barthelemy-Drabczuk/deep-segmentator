"""Pytest configuration and shared fixtures."""
import pytest

from sulcal_seg.data.dataset_manager import DatasetManager
from tests.fixtures.mock_configs import make_eas_config, make_gan_config, make_laplace_config


@pytest.fixture(autouse=True)
def reset_dataset_manager():
    """Reset the DatasetManager singleton before and after each test."""
    DatasetManager.reset()
    yield
    DatasetManager.reset()


@pytest.fixture
def eas_config():
    """Minimal valid EAS config dict."""
    return make_eas_config()


@pytest.fixture
def gan_config():
    """Minimal valid GAN config dict."""
    return make_gan_config()


@pytest.fixture
def laplace_config():
    """Minimal valid Laplace config dict."""
    return make_laplace_config()


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
