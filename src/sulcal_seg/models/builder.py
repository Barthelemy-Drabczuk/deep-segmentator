"""ModelBuilder — fluent API for assembling sulcal segmentation model variants.

Usage example::

    from sulcal_seg.models.builder import ModelBuilder

    model = (
        ModelBuilder()
        .with_eas(eas_config)
        .with_gan(gan_config)
        .with_laplace(laplace_config)
        .build()
    )

All 7 ablation configurations (any subset of EAS / GAN / Laplace) are valid,
including the baseline with no components.
"""
from typing import Any, Dict, Optional

from sulcal_seg.models.components.eas.eas_engine import EASEngine
from sulcal_seg.models.components.gan.gan_engine import GANEngine
from sulcal_seg.models.components.laplace.laplace_engine import LaplaceEngine
from sulcal_seg.models.full_model import SulcalSegmentationModel


class ModelBuilder:
    """
    Fluent builder for :class:`SulcalSegmentationModel`.

    Methods return `self` to support method chaining.
    Call :meth:`build` to instantiate the model.
    """

    def __init__(self) -> None:
        self._eas_config: Optional[Dict[str, Any]] = None
        self._gan_config: Optional[Dict[str, Any]] = None
        self._laplace_config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Component configuration methods
    # ------------------------------------------------------------------

    def with_eas(self, config: Dict[str, Any]) -> "ModelBuilder":
        """
        Configure the EAS (Evolutionary Architecture Search) component.

        Args:
            config: Dict matching EASComponentConfig fields.
                    Must contain at least: 'population_size', 'num_generations',
                    'num_classes'.

        Returns:
            self (for chaining)

        Raises:
            ValueError: If required keys are missing from config.
        """
        _REQUIRED = {"population_size", "num_generations", "num_classes"}
        missing = _REQUIRED - set(config.keys())
        if missing:
            raise ValueError(f"EAS config missing keys: {missing}")
        self._eas_config = config
        return self

    def with_gan(self, config: Dict[str, Any]) -> "ModelBuilder":
        """
        Configure the GAN segmentation component.

        Args:
            config: Dict matching GANComponentConfig fields.
                    Must contain at least: 'num_classes', 'in_channels'.

        Returns:
            self (for chaining)

        Raises:
            ValueError: If required keys are missing from config.
        """
        _REQUIRED = {"num_classes", "in_channels"}
        missing = _REQUIRED - set(config.keys())
        if missing:
            raise ValueError(f"GAN config missing keys: {missing}")
        self._gan_config = config
        return self

    def with_laplace(self, config: Dict[str, Any]) -> "ModelBuilder":
        """
        Configure the Laplace geometry constraint component.

        Args:
            config: Dict matching LaplaceComponentConfig fields.
                    Must contain at least: 'num_iterations', 'lambda_smooth'.

        Returns:
            self (for chaining)

        Raises:
            ValueError: If required keys are missing from config.
        """
        _REQUIRED = {"num_iterations", "lambda_smooth"}
        missing = _REQUIRED - set(config.keys())
        if missing:
            raise ValueError(f"Laplace config missing keys: {missing}")
        self._laplace_config = config
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> SulcalSegmentationModel:
        """
        Instantiate and return a :class:`SulcalSegmentationModel`.

        Builds only the components that have been configured via
        `with_eas`, `with_gan`, and/or `with_laplace`.

        Returns:
            Assembled :class:`SulcalSegmentationModel`.
        """
        eas: Optional[EASEngine] = None
        gan: Optional[GANEngine] = None
        laplace: Optional[LaplaceEngine] = None

        if self._eas_config is not None:
            eas = EASEngine(self._eas_config)

        if self._gan_config is not None:
            gan = GANEngine(self._gan_config)

        if self._laplace_config is not None:
            laplace = LaplaceEngine(self._laplace_config)

        return SulcalSegmentationModel(eas=eas, gan=gan, laplace=laplace)

    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_all_ablations(
        eas_config: Dict[str, Any],
        gan_config: Dict[str, Any],
        laplace_config: Dict[str, Any],
    ) -> Dict[str, SulcalSegmentationModel]:
        """
        Build all 7 non-trivial ablation configurations.

        Args:
            eas_config: Config dict for EAS component.
            gan_config: Config dict for GAN component.
            laplace_config: Config dict for Laplace component.

        Returns:
            Dict mapping ablation name → assembled model.
        """
        configs = {
            "eas_only": ModelBuilder().with_eas(eas_config).build(),
            "gan_only": ModelBuilder().with_gan(gan_config).build(),
            "laplace_only": ModelBuilder().with_laplace(laplace_config).build(),
            "eas_gan": ModelBuilder().with_eas(eas_config).with_gan(gan_config).build(),
            "eas_laplace": (
                ModelBuilder().with_eas(eas_config).with_laplace(laplace_config).build()
            ),
            "gan_laplace": (
                ModelBuilder().with_gan(gan_config).with_laplace(laplace_config).build()
            ),
            "full": (
                ModelBuilder()
                .with_eas(eas_config)
                .with_gan(gan_config)
                .with_laplace(laplace_config)
                .build()
            ),
        }
        return configs
