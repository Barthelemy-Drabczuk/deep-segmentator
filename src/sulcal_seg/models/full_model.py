"""Full sulcal segmentation model combining EAS + GAN + Laplace components."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sulcal_seg.models.components.eas.eas_engine import EASEngine
from sulcal_seg.models.components.gan.gan_engine import GANEngine
from sulcal_seg.models.components.laplace.laplace_engine import LaplaceEngine


class SulcalSegmentationModel(nn.Module):
    """
    Full lifespan-robust sulcal segmentation model.

    Optionally assembles up to three components:
        - EAS: Evolutionary Architecture Search (finds optimal U-Net)
        - GAN: Multi-label adversarial training (prevents mode collapse)
        - Laplace: Geometry post-processing (topologically valid meshes)

    All 7 ablation configurations are valid:
        1. baseline (none)
        2. EAS only
        3. GAN only
        4. Laplace only
        5. EAS + GAN
        6. EAS + Laplace
        7. GAN + Laplace
        8. EAS + GAN + Laplace (full model)

    Args:
        eas: Optional EASEngine instance.
        gan: Optional GANEngine instance.
        laplace: Optional LaplaceEngine instance.
    """

    def __init__(
        self,
        eas: Optional[EASEngine] = None,
        gan: Optional[GANEngine] = None,
        laplace: Optional[LaplaceEngine] = None,
    ) -> None:
        super().__init__()

        self.eas = eas
        self.gan = gan
        self.laplace = laplace

        # Register nn.Module components so parameters are tracked
        if eas is not None:
            # EASEngine is not an nn.Module — wrap its generator if available
            pass
        if gan is not None:
            self.add_module("_gan_generator", gan.generator)
            self.add_module("_gan_discriminator", gan.discriminator)

    @property
    def has_eas(self) -> bool:
        return self.eas is not None

    @property
    def has_gan(self) -> bool:
        return self.gan is not None

    @property
    def has_laplace(self) -> bool:
        return self.laplace is not None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """
        Run the assembled pipeline on input MRI volume `x`.

        Pipeline (present components only):
            GAN → Laplace (EAS provides the architecture used by GAN)

        Args:
            x: Input MRI batch (B, C, D, H, W).

        Returns:
            Segmentation logits or post-processed label map.
        """
        if not self.has_gan and not self.has_eas:
            return self._baseline_forward(x)

        if self.has_gan:
            output = self.gan.forward(x)
        else:
            # EAS-only: use best discovered architecture
            output = self.eas.forward(x)

        if self.has_laplace:
            # Laplace post-processing (stub)
            probs = torch.softmax(output, dim=1).detach().cpu().numpy()
            output = self.laplace.forward(probs)

        return output

    def _baseline_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Baseline forward pass (no learned components assembled).

        Returns a zero-filled logit map — placeholder for testing the
        builder pattern without GPU or real weights.

        Args:
            x: (B, C, D, H, W) input tensor.

        Returns:
            Zero tensor (B, 1, D, H, W).
        """
        B = x.shape[0]
        spatial = x.shape[2:]  # (D, H, W)
        return torch.zeros(B, 1, *spatial, device=x.device)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        components = []
        if self.has_eas:
            components.append("EAS")
        if self.has_gan:
            components.append("GAN")
        if self.has_laplace:
            components.append("Laplace")
        active = "+".join(components) if components else "baseline"
        return f"SulcalSegmentationModel({active})"

    def component_summary(self) -> Dict[str, Any]:
        """Return a summary dict of which components are active."""
        return {
            "has_eas": self.has_eas,
            "has_gan": self.has_gan,
            "has_laplace": self.has_laplace,
            "active_components": [
                name
                for name, flag in [
                    ("eas", self.has_eas),
                    ("gan", self.has_gan),
                    ("laplace", self.has_laplace),
                ]
                if flag
            ],
        }
