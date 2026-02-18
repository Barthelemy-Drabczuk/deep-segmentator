"""HPO search space definitions for Optuna."""
from typing import Any, Dict


class HyperparameterSpace:
    """
    Defines the hyperparameter search space for each component.

    Each `suggest_*` method reads from an Optuna ``Trial`` object
    and returns a config dict suitable for the corresponding component.
    """

    def suggest_eas_params(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest EAS component hyperparameters.

        Args:
            trial: optuna.Trial object.

        Returns:
            EASComponentConfig-compatible dict.
        """
        return {
            "population_size": trial.suggest_int("eas_population_size", 10, 50),
            "num_generations": trial.suggest_int("eas_num_generations", 10, 100),
            "num_classes": 41,
            "min_filters": trial.suggest_categorical("eas_min_filters", [8, 16]),
            "max_filters": trial.suggest_categorical("eas_max_filters", [128, 256, 512]),
            "min_depth": trial.suggest_int("eas_min_depth", 2, 3),
            "max_depth": trial.suggest_int("eas_max_depth", 4, 6),
            "mutation_rate": trial.suggest_float("eas_mutation_rate", 0.1, 0.5),
            "hosvd_rank_ratios": [
                trial.suggest_float("hosvd_rank_0", 0.5, 1.0),
                trial.suggest_float("hosvd_rank_1", 0.5, 1.0),
                trial.suggest_float("hosvd_rank_2", 0.5, 1.0),
            ],
        }

    def suggest_gan_params(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest GAN component hyperparameters.

        Args:
            trial: optuna.Trial object.

        Returns:
            GANComponentConfig-compatible dict.
        """
        return {
            "num_classes": 41,
            "in_channels": 1,
            "base_filters": trial.suggest_categorical("gan_base_filters", [16, 32, 64]),
            "depth": trial.suggest_int("gan_depth", 3, 5),
            "lambda_dice": trial.suggest_float("gan_lambda_dice", 0.5, 2.0),
            "lambda_gan": trial.suggest_float("gan_lambda_gan", 0.05, 0.5),
            "lambda_laplace": trial.suggest_float("gan_lambda_laplace", 0.01, 0.2),
            "learning_rate_g": trial.suggest_float("gan_lr_g", 1e-5, 1e-3, log=True),
            "learning_rate_d": trial.suggest_float("gan_lr_d", 1e-5, 1e-3, log=True),
            "gan_mode": trial.suggest_categorical("gan_mode", ["lsgan", "vanilla"]),
        }

    def suggest_laplace_params(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest Laplace component hyperparameters.

        Args:
            trial: optuna.Trial object.

        Returns:
            LaplaceComponentConfig-compatible dict.
        """
        return {
            "num_iterations": trial.suggest_int("laplace_num_iter", 10, 100),
            "lambda_smooth": trial.suggest_float("laplace_lambda", 0.2, 0.8),
            "algorithm": trial.suggest_categorical("laplace_algo", ["taubin", "laplacian"]),
        }

    def suggest_all_params(self, trial: Any) -> Dict[str, Dict[str, Any]]:
        """
        Suggest hyperparameters for all three components simultaneously.

        Args:
            trial: optuna.Trial object.

        Returns:
            Dict with keys 'eas', 'gan', 'laplace'.
        """
        return {
            "eas": self.suggest_eas_params(trial),
            "gan": self.suggest_gan_params(trial),
            "laplace": self.suggest_laplace_params(trial),
        }
