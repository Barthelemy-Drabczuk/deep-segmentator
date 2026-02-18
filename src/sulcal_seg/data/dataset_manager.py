"""
Dataset management: Singleton DatasetManager + DatasetLoader factory.

Load data ONCE, cache for all components.
Thread-safe via double-checked locking.
"""
import threading
from pathlib import Path
from typing import Dict, List, Optional, Type

from .abstract_loader import AbstractDataLoader
from .loaders.abcd_loader import ABCDLoader
from .loaders.abide_loader import ABIDELoader
from .loaders.custom_loader import CustomLoader
from .loaders.senior_loader import SENIORLoader
from .loaders.ukbiobank_loader import UKBiobankLoader


class DatasetLoader:
    """
    Factory for creating dataset loaders by name.

    Usage:
        loader = DatasetLoader.get_loader('ukbiobank', root_dir=Path('/data/ukbiobank'), subset='train')
        # Register custom loader at runtime:
        DatasetLoader.register('my_dataset', MyDatasetLoader)
        loader = DatasetLoader.get_loader('my_dataset', root_dir=Path('/data/mine'), subset='train')
    """

    LOADERS: Dict[str, Type[AbstractDataLoader]] = {
        "ukbiobank": UKBiobankLoader,
        "abcd": ABCDLoader,
        "abide": ABIDELoader,
        "senior": SENIORLoader,
        "custom": CustomLoader,
    }

    @staticmethod
    def get_loader(dataset_name: str, **kwargs) -> AbstractDataLoader:
        """
        Instantiate a dataset loader by name.

        Args:
            dataset_name: One of 'ukbiobank', 'abcd', 'abide', 'senior', 'custom'
            **kwargs: Arguments forwarded to the loader constructor (e.g., root_dir, subset)

        Returns:
            Instantiated AbstractDataLoader subclass
        """
        name = dataset_name.lower()
        if name not in DatasetLoader.LOADERS:
            known = list(DatasetLoader.LOADERS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name!r}. Known datasets: {known}")
        return DatasetLoader.LOADERS[name](**kwargs)

    @staticmethod
    def register(name: str, loader_class: Type[AbstractDataLoader]) -> None:
        """
        Register a custom dataset loader.

        Args:
            name: Name to register the loader under
            loader_class: Class that inherits from AbstractDataLoader
        """
        DatasetLoader.LOADERS[name.lower()] = loader_class


class DatasetManager:
    """
    Thread-safe Singleton for managing dataset loaders.

    Load data ONCE per session, serve to all components.

    Usage:
        manager = DatasetManager.get_instance()
        manager.register_dataset('train', 'ukbiobank', Path('/data/ukbiobank'), subset='train')
        subject_data = manager.get_subject('train', 'sub-001')
        # Returns: {'subject_id': ..., 'image': ..., 'morphologist_label': ..., ...}
    """

    _instance: Optional["DatasetManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DatasetManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self.loaders: Dict[str, AbstractDataLoader] = {}
        self._cache: Dict[tuple, Dict] = {}
        self._initialized = True

    def register_dataset(
        self,
        dataset_name: str,
        loader_class: str,
        root_dir: Path,
        **loader_kwargs,
    ) -> None:
        """
        Register and instantiate a dataset loader.

        Args:
            dataset_name: Logical name for this dataset split (e.g., 'train', 'ukbiobank_val')
            loader_class: Dataset type string (e.g., 'ukbiobank', 'abcd')
            root_dir: Root directory of the dataset
            **loader_kwargs: Additional kwargs for the loader (e.g., subset='train')
        """
        loader = DatasetLoader.get_loader(loader_class, root_dir=root_dir, **loader_kwargs)
        self.loaders[dataset_name] = loader

    def get_subject(self, dataset_name: str, subject_id: str) -> Dict:
        """
        Get subject data (cached to avoid reloading).

        Args:
            dataset_name: Registered dataset name (e.g., 'train')
            subject_id: Subject identifier (e.g., 'sub-001')

        Returns:
            Dict with 'subject_id', 'image', 'morphologist_label', 'freesurfer_label', 'metadata'
        """
        cache_key = (dataset_name, subject_id)
        if cache_key not in self._cache:
            if dataset_name not in self.loaders:
                raise ValueError(
                    f"Dataset not registered: {dataset_name!r}. "
                    f"Available: {list(self.loaders.keys())}"
                )
            loader = self.loaders[dataset_name]
            subject_ids = loader.get_subject_ids()
            try:
                idx = subject_ids.index(subject_id)
            except ValueError:
                raise ValueError(
                    f"Subject {subject_id!r} not found in dataset {dataset_name!r}"
                )
            self._cache[cache_key] = loader[idx]
        return self._cache[cache_key]

    def get_batch(self, dataset_name: str, subject_ids: List[str]) -> List[Dict]:
        """Get multiple subjects as a list of dicts."""
        return [self.get_subject(dataset_name, sid) for sid in subject_ids]

    def clear_cache(self) -> None:
        """Clear the subject cache (useful when switching datasets)."""
        self._cache.clear()

    @classmethod
    def get_instance(cls) -> "DatasetManager":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton (for testing only).

        This allows tests to start with a fresh DatasetManager state.
        """
        with cls._lock:
            cls._instance = None
