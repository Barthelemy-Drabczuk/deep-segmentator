"""Optuna objective functions (stubs — require GPU + data)."""
from typing import Any, Callable, Dict, Optional

from sulcal_seg.hpo.search_space import HyperparameterSpace
from sulcal_seg.models.builder import ModelBuilder


class TrainingObjectives:
    """
    Collection of Optuna objective callables for different training setups.

    Each objective returns a scalar validation metric (to be maximised or
    minimised depending on the study direction).
    """

    def __init__(
        self,
        train_loader: Any,
        val_loader: Any,
        base_config: Dict[str, Any],
    ) -> None:
        """
        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            base_config: Base training configuration dict.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = base_config
        self.space = HyperparameterSpace()

    def objective_full_model(self, trial: Any) -> float:
        """
        Objective for full EAS + GAN + Laplace model.

        Builds the model from Optuna-suggested hyperparameters, trains it
        briefly, and returns the validation macro-Dice score.

        Args:
            trial: optuna.Trial object.

        Returns:
            Validation macro-Dice (higher = better).

        Raises:
            NotImplementedError: Always (stub — requires GPU + data).
        """
        # TODO: 1. params = self.space.suggest_all_params(trial)
        #   2. model = ModelBuilder()
        #              .with_eas(params['eas'])
        #              .with_gan(params['gan'])
        #              .with_laplace(params['laplace'])
        #              .build()
        #   3. trainer = Trainer(model, self.base_config)
        #      trainer.add_optuna_callback(trial)
        #   4. metrics = trainer.train(self.train_loader, self.val_loader)
        #   5. return metrics['val_macro_dice']
        raise NotImplementedError(
            "objective_full_model() requires GPU and real data loaders."
        )

    def objective_gan_only(self, trial: Any) -> float:
        """
        Objective for GAN-only ablation study.

        Raises:
            NotImplementedError: Always (stub).
        """
        raise NotImplementedError(
            "objective_gan_only() requires GPU and real data loaders."
        )
