"""U-Net generator for the GAN segmentation component."""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    3D convolution block: Conv3d → BN → Activation (×2).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        activation: Activation name ('relu', 'leaky_relu', 'elu', 'gelu').
        dropout: Dropout probability applied after activation (0 = off).
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act = self._ACTIVATIONS.get(activation, nn.ReLU(inplace=True))
        layers: List[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act,
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    3D U-Net generator for sulcal segmentation.

    Produces a multi-class probability map from a T1 MRI volume.

    Architecture:
        - Encoder: `depth` levels of ConvBlock3D + MaxPool3d downsampling
        - Bottleneck: ConvBlock3D at the lowest resolution
        - Decoder: `depth` levels of bilinear up-sampling + skip concatenation
        - Head: 1×1×1 Conv to `num_classes` logits

    Note:
        The current implementation uses `depth=2` as a functional skeleton.
        TODO: Extend to full depth≥4 with parameterised filter counts.

    Args:
        in_channels: Number of input channels (1 for T1 MRI).
        num_classes: Number of output segmentation classes.
        base_filters: Filters at the first encoder level.
        depth: Number of encoder/decoder levels.
        activation: Activation function name.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 41,
        base_filters: int = 32,
        depth: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes

        # --- Encoder ---
        self.enc1 = ConvBlock3D(in_channels, base_filters, activation)
        self.enc2 = ConvBlock3D(base_filters, base_filters * 2, activation)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock3D(base_filters * 2, base_filters * 4, activation)

        # --- Decoder ---
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_filters * 4, base_filters * 2, activation)  # + skip

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_filters * 2, base_filters, activation)  # + skip

        # --- Output head ---
        self.head = nn.Conv3d(base_filters, num_classes, kernel_size=1)

        # TODO: Extend encoder/decoder to `depth` levels using the full
        #       filter schedule [base, base*2, base*4, ..., base*2^depth].

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, D, H, W).

        Returns:
            Logit tensor of shape (B, num_classes, D, H, W).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (for inference / mode-collapse detection)."""
        return torch.softmax(self.forward(x), dim=1)
