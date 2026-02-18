"""Optuna HPO manager for sulcal segmentation."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from sulcal_seg.hpo.pruning import create_hyperband_pruner


class OptunaHPOManager:
    """
    Manages an Optuna hyperparameter optimisation study.

    Features:
        - TPE sampler (Tree-structured Parzen Estimator)
        - Hyperband pruner for early stopping of unpromising trials
        - SQLite-backed persistent storage (resumable across SLURM jobs)
        - CSV + YAML result export

    Args:
        study_name: Unique name for this optimisation study.
        storage_path: Path to SQLite database file.
        n_trials: Number of trials to run.
        direction: 'maximize' or 'minimize'.
        n_jobs: Parallel workers (-1 = all available).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        study_name: str,
        storage_path: str = "hpo_results.db",
        n_trials: int = 100,
        direction: str = "maximize",
        n_jobs: int = 1,
        seed: Optional[int] = 42,
    ) -> None:
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self.direction = direction
        self.n_jobs = n_jobs
        self.seed = seed
        self._study: Optional[Any] = None

    def _create_or_load_study(self) -> Any:
        """Create a new study or resume from existing SQLite storage."""
        import optuna  # noqa: PLC0415

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = create_hyperband_pruner(min_resource=5, max_resource=50)
        storage_url = f"sqlite:///{self.storage_path}"

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        return study

    def optimize(self, objective: Callable[[Any], float]) -> Any:
        """
        Run the optimisation loop.

        Args:
            objective: Callable(trial) → scalar metric.

        Returns:
            The completed optuna.Study object.
        """
        self._study = self._create_or_load_study()
        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        return self._study

    def save_results(self, output_dir: str) -> Dict[str, Path]:
        """
        Export study results to YAML and CSV.

        Args:
            output_dir: Directory to write output files.

        Returns:
            Dict mapping 'yaml' and 'csv' to their output paths.
        """
        import pandas as pd  # noqa: PLC0415

        if self._study is None:
            raise RuntimeError("Call optimize() before save_results().")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Best params → YAML
        yaml_path = out / "best_params.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(
                self._study.best_trial.params,
                f,
                default_flow_style=False,
                sort_keys=True,
            )

        # All trials → CSV
        csv_path = out / "all_trials.csv"
        df = self._study.trials_dataframe()
        df.to_csv(csv_path, index=False)

        return {"yaml": yaml_path, "csv": csv_path}

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best trial's hyperparameters."""
        if self._study is None:
            self._study = self._create_or_load_study()
        return self._study.best_trial.params
