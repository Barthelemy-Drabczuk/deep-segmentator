"""U-Net generator for the GAN segmentation component."""
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Production architecture: ResNet-backed 3D U-Net
# ---------------------------------------------------------------------------


class ResNetBlock3D(nn.Module):
    """
    3D residual block (BasicBlock-style).

    Two 3×3×3 convolutions with batch normalisation and a residual shortcut.
    The shortcut is a 1×1×1 projection when channels or stride change.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Stride for the first convolution (use 2 for downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class UNet3DResNet(nn.Module):
    """
    3D U-Net with ResNet backbone for sulcal segmentation (production model).

    5-level encoder-decoder with skip connections. Designed to process
    256³ MRI volumes; minimum spatial size is 16³ (4 stride-2 encoder levels).

    Architecture:
        Encoder  : enc0 (stride-1) + enc1–enc4 (stride-2 each)
        Bottleneck: 2 × ResNetBlock3D
        Decoder  : 4 upconv stages; each concatenates enc(N-1) skip then decodes
        Head     : 1×1×1 Conv3d → num_classes logits

    Skip connection pairing (correct — avoids off-by-one):
        dec4 ← cat(upconv4(bottleneck), enc3)
        dec3 ← cat(upconv3(dec4),       enc2)
        dec2 ← cat(upconv2(dec3),       enc1)
        dec1 ← cat(upconv1(dec2),       enc0)

    Args:
        in_channels: MRI input channels (1 for T1).
        num_classes: Segmentation output classes (52 sulcal labels).
        base_channels: Filter count at enc0; doubles each encoder level.
        resnet_depth: ResNet variant (18, 34, or 50). Selects blocks per level.
    """

    # Number of ResNetBlock3D per encoder level for each depth variant
    _BLOCKS_PER_LEVEL: Dict[int, List[int]] = {
        18: [2, 2, 2, 2, 2],
        34: [3, 4, 6, 3, 3],
        50: [3, 4, 6, 3, 3],
    }

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 52,
        base_channels: int = 64,
        resnet_depth: int = 18,
    ) -> None:
        super().__init__()
        c = base_channels
        n = self._BLOCKS_PER_LEVEL.get(resnet_depth, self._BLOCKS_PER_LEVEL[18])

        # Encoder
        self.enc0 = self._make_level(in_channels, c,      n[0], stride=1)
        self.enc1 = self._make_level(c,           c * 2,  n[1], stride=2)
        self.enc2 = self._make_level(c * 2,       c * 4,  n[2], stride=2)
        self.enc3 = self._make_level(c * 4,       c * 8,  n[3], stride=2)
        self.enc4 = self._make_level(c * 8,       c * 16, n[4], stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResNetBlock3D(c * 16, c * 16),
            ResNetBlock3D(c * 16, c * 16),
        )

        # Decoder — upconv halves channels; concat adds skip; dec block fuses
        self.upconv4 = nn.ConvTranspose3d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ResNetBlock3D(c * 16, c * 8)   # (c*8 up + c*8 enc3) → c*8

        self.upconv3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ResNetBlock3D(c * 8, c * 4)    # (c*4 up + c*4 enc2) → c*4

        self.upconv2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ResNetBlock3D(c * 4, c * 2)    # (c*2 up + c*2 enc1) → c*2

        self.upconv1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ResNetBlock3D(c * 2, c)         # (c up + c enc0) → c

        self.head = nn.Conv3d(c, num_classes, kernel_size=1)

    @staticmethod
    def _make_level(
        in_ch: int, out_ch: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build one encoder level: first block applies stride, rest are stride-1."""
        blocks: List[nn.Module] = [ResNetBlock3D(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResNetBlock3D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input MRI volume (B, in_channels, D, H, W). Minimum size: 16³.

        Returns:
            Segmentation logits (B, num_classes, D, H, W).
        """
        e0 = self.enc0(x)     # (B, c,    D,    H,    W)
        e1 = self.enc1(e0)    # (B, c*2,  D/2,  H/2,  W/2)
        e2 = self.enc2(e1)    # (B, c*4,  D/4,  H/4,  W/4)
        e3 = self.enc3(e2)    # (B, c*8,  D/8,  H/8,  W/8)
        e4 = self.enc4(e3)    # (B, c*16, D/16, H/16, W/16)

        b = self.bottleneck(e4)  # (B, c*16, D/16, H/16, W/16)

        # Decoder: pair each upconv output with enc(N-1) skip
        d4 = self.dec4(torch.cat([self.upconv4(b),   e3], dim=1))  # D/8
        d3 = self.dec3(torch.cat([self.upconv3(d4),  e2], dim=1))  # D/4
        d2 = self.dec2(torch.cat([self.upconv2(d3),  e1], dim=1))  # D/2
        d1 = self.dec1(torch.cat([self.upconv1(d2),  e0], dim=1))  # D

        return self.head(d1)  # (B, num_classes, D, H, W)
