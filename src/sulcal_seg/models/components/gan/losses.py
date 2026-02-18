"""Loss functions for the GAN segmentation component.

Includes:
- DiceLoss: Multi-class differentiable Dice for segmentation.
- GANLoss: Adversarial loss (vanilla, LSGAN, WGAN).
- LaplaceLoss: Geometry regularisation via finite-difference Laplacian.
- CombinedLoss: Weighted sum of all three.
"""
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    L = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)

    averaged over classes (macro Dice).

    Args:
        smooth: Laplace smoothing term to avoid division by zero.
        ignore_index: Class index to exclude from the average (-1 = include all).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, C, D, H, W) or probabilities.
            target: Integer class labels (B, D, H, W) or one-hot (B, C, D, H, W).

        Returns:
            Scalar Dice loss.
        """
        n_classes = pred.shape[1]
        probs = torch.softmax(pred, dim=1)  # (B, C, D, H, W)

        # Convert integer targets to one-hot
        if target.dim() == pred.dim() - 1:
            one_hot = F.one_hot(target.long(), num_classes=n_classes)  # (B, D, H, W, C)
            one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        else:
            one_hot = target.float()

        dice_per_class = []
        for c in range(n_classes):
            if c == self.ignore_index:
                continue
            p = probs[:, c].contiguous().view(-1)
            t = one_hot[:, c].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(dice)

        return 1.0 - torch.stack(dice_per_class).mean()


class GANLoss(nn.Module):
    """
    Adversarial loss for the GAN discriminator.

    Supports three modes:
        - 'vanilla': Binary cross-entropy (original GAN)
        - 'lsgan': Least-squares (LSGAN) — more stable gradients
        - 'wgan': Wasserstein loss (use with gradient penalty externally)

    Args:
        mode: GAN loss variant.
        real_label: Target value for real samples (label smoothing applies here).
        fake_label: Target value for fake samples.
    """

    def __init__(
        self,
        mode: Literal["vanilla", "lsgan", "wgan"] = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.real_label = real_label
        self.fake_label = fake_label

    def _get_target(
        self, pred: torch.Tensor, is_real: bool
    ) -> torch.Tensor:
        value = self.real_label if is_real else self.fake_label
        return torch.full_like(pred, value)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Args:
            pred: Discriminator output (B, 1).
            is_real: True for real samples, False for generated.

        Returns:
            Scalar loss.
        """
        if self.mode == "vanilla":
            target = self._get_target(pred, is_real)
            return F.binary_cross_entropy_with_logits(pred, target)
        elif self.mode == "lsgan":
            target = self._get_target(pred, is_real)
            return F.mse_loss(pred, target)
        elif self.mode == "wgan":
            # WGAN: maximise E[D(real)] - E[D(fake)]
            # For discriminator real: -mean(pred); for discriminator fake: +mean(pred)
            # For generator: -mean(pred)
            return -pred.mean() if is_real else pred.mean()
        else:
            raise ValueError(f"Unknown GAN mode: {self.mode}")


class LaplaceLoss(nn.Module):
    """
    Laplacian smoothness regularisation for 3-D segmentation probability maps.

    Penalises high-frequency spatial variation by computing the magnitude of
    the discrete 3D Laplacian of the predicted probability map.

    L = mean(|∇²P|²)

    where ∇² is the finite-difference Laplacian operator.

    Args:
        weight: Loss weight multiplier.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        # 3D Laplacian kernel (6-connectivity)
        kernel = torch.zeros(1, 1, 3, 3, 3)
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits or probabilities (B, C, D, H, W).

        Returns:
            Scalar Laplacian smoothness loss.
        """
        probs = torch.softmax(pred, dim=1)  # (B, C, D, H, W)
        B, C, D, H, W = probs.shape

        # Apply Laplacian independently per class
        probs_flat = probs.view(B * C, 1, D, H, W)
        kernel = self.kernel.to(probs.device)
        laplacian = F.conv3d(probs_flat, kernel, padding=1)
        return self.weight * (laplacian ** 2).mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of DiceLoss + GANLoss + LaplaceLoss.

    L = λ_dice * L_dice + λ_gan * L_gan + λ_laplace * L_laplace

    Args:
        lambda_dice: Weight for Dice loss.
        lambda_gan: Weight for GAN adversarial loss.
        lambda_laplace: Weight for Laplacian smoothness loss.
        gan_mode: GAN loss mode passed to GANLoss.
        smooth: Dice smoothing term.
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_gan: float = 0.1,
        lambda_laplace: float = 0.05,
        gan_mode: Literal["vanilla", "lsgan", "wgan"] = "lsgan",
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_gan = lambda_gan
        self.lambda_laplace = lambda_laplace

        self.dice_loss = DiceLoss(smooth=smooth)
        self.gan_loss = GANLoss(mode=gan_mode)
        self.laplace_loss = LaplaceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Generator logits (B, C, D, H, W).
            target: Integer labels (B, D, H, W).
            disc_pred: Discriminator output on generated samples (B, 1).
                       If None, GAN term is skipped.

        Returns:
            Scalar combined loss.
        """
        loss = self.lambda_dice * self.dice_loss(pred, target)
        loss = loss + self.lambda_laplace * self.laplace_loss(pred)
        if disc_pred is not None and self.lambda_gan > 0.0:
            # Generator wants discriminator to classify fakes as real
            loss = loss + self.lambda_gan * self.gan_loss(disc_pred, is_real=True)
        return loss
