"""Patch discriminator for the GAN segmentation component."""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    3D patch discriminator that classifies segmentation volumes as real
    or fake.

    Takes a concatenation of [image, segmentation_label] as input and
    outputs a scalar "realness" score per spatial patch.

    Trained on BOTH Morphologist AND FreeSurfer labels simultaneously
    to prevent mode collapse towards a single labelling convention.

    Architecture:
        - 2 convolutional layers with LeakyReLU + spectral normalisation
        - Average pooling + linear head
        TODO: Extend to full PatchGAN depth (4–5 conv layers, no global pool).

    Args:
        in_channels: Input channels = image_channels + num_classes.
        base_filters: Filters in first conv layer.
    """

    def __init__(self, in_channels: int = 42, base_filters: int = 64) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1
            nn.utils.spectral_norm(
                nn.Conv3d(
                    in_channels, base_filters, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.utils.spectral_norm(
                nn.Conv3d(
                    base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.BatchNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # TODO: Add conv layers 3–5 for full PatchGAN depth.
        )

        self.global_avg = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.head = nn.Linear(base_filters * 2, 1)

    def forward(
        self,
        image: torch.Tensor,
        segmentation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image: T1 volume (B, 1, D, H, W).
            segmentation: Probability map or one-hot (B, num_classes, D, H, W).

        Returns:
            Realness score per batch element (B, 1).
        """
        # (B, 1+num_classes, D, H, W)
        x = torch.cat([image, segmentation], dim=1)
        features = self.features(x)
        pooled = self.global_avg(features).view(features.size(0), -1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Production discriminator: segmentation-only for dual-label GAN
# ---------------------------------------------------------------------------


class SulcalDiscriminator3D(nn.Module):
    """
    3D discriminator for the dual-label sulcal GAN.

    Takes only a segmentation probability map as input (no image concat).
    Designed to classify 52-class sulcal maps as real (Morphologist or
    FreeSurfer labels) or fake (generator output), preventing mode collapse
    by being trained on two real label sources simultaneously.

    Architecture:
        4 conv blocks (spectral-norm Conv3d stride-2 + BN + LeakyReLU)
        → AdaptiveAvgPool3d → Linear → (B, 1)

    Args:
        in_channels: Segmentation input channels (52 sulcal classes).
        base_channels: Filter count of the first conv layer.
    """

    def __init__(self, in_channels: int = 52, base_channels: int = 64) -> None:
        super().__init__()

        def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv3d(
                        in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
                    )
                ),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        c = base_channels
        self.features = nn.Sequential(
            _conv_block(in_channels, c),       # → c
            _conv_block(c,           c * 2),   # → c*2
            _conv_block(c * 2,       c * 4),   # → c*4
            _conv_block(c * 4,       c * 8),   # → c*8
        )
        self.global_avg = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.head = nn.Linear(c * 8, 1)

    def forward(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segmentation: Probability map or one-hot (B, 52, D, H, W).

        Returns:
            Realness score per batch element (B, 1).
        """
        features = self.features(segmentation)
        pooled = self.global_avg(features).view(features.size(0), -1)
        return self.head(pooled)
