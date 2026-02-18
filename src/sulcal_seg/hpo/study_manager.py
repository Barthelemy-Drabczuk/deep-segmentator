"""Optuna study persistence and best-params export."""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class StudyManager:
    """
    Loads, saves, and exports Optuna studies.

    Provides utilities for:
        - Loading an existing study from SQLite storage.
        - Exporting the best trial's parameters to a YAML file.
        - Summarising study results.
    """

    def __init__(self, storage_path: str, study_name: str) -> None:
        """
        Args:
            storage_path: Path to the SQLite database file.
            study_name: Name of the Optuna study.
        """
        self.storage_path = storage_path
        self.study_name = study_name

    def load_study(self) -> Any:
        """
        Load an existing Optuna study from SQLite storage.

        Returns:
            optuna.Study instance.
        """
        import optuna  # noqa: PLC0415

        storage_url = f"sqlite:///{self.storage_path}"
        return optuna.load_study(study_name=self.study_name, storage=storage_url)

    def get_best_params(self) -> Dict[str, Any]:
        """
        Return the hyperparameters of the best trial.

        Returns:
            Dict of best hyperparameter values.
        """
        study = self.load_study()
        return study.best_trial.params

    def export_best_params(self, output_yaml: str) -> Path:
        """
        Write the best trial's parameters to a YAML file.

        Args:
            output_yaml: Path to output YAML file.

        Returns:
            Path to the written file.
        """
        params = self.get_best_params()
        output_path = Path(output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=True)
        return output_path

    def summary(self) -> Dict[str, Any]:
        """
        Return a human-readable summary of the study.

        Returns:
            Dict with 'n_trials', 'best_value', 'best_params', 'directions'.
        """
        study = self.load_study()
        best = study.best_trial
        return {
            "n_trials": len(study.trials),
            "n_complete": sum(
                1 for t in study.trials
                if t.state.name == "COMPLETE"
            ),
            "best_value": best.value,
            "best_params": best.params,
            "study_name": study.study_name,
        }
