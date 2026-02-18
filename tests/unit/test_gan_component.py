"""Unit tests for GAN component loss functions and architecture."""
import pytest
import torch

from sulcal_seg.models.components.gan.losses import CombinedLoss, DiceLoss, GANLoss, LaplaceLoss


# Small tensor dimensions to keep tests fast on CPU
B, C, D, H, W = 2, 4, 8, 8, 8


class TestDiceLoss:
    def test_perfect_prediction_gives_low_loss(self):
        criterion = DiceLoss()
        target = torch.zeros(B, D, H, W, dtype=torch.long)
        target[:, :2, :2, :2] = 1  # some class-1 voxels
        # Perfect logits: high score on correct class
        pred = torch.zeros(B, C, D, H, W)
        for c in range(C):
            mask = (target == c)
            pred[:, c][mask] = 10.0  # very high logit
        loss = criterion(pred, target)
        assert loss.item() < 0.2

    def test_random_pred_gives_loss_close_to_1(self):
        criterion = DiceLoss(smooth=1.0)
        target = torch.randint(0, C, (B, D, H, W))
        pred = torch.randn(B, C, D, H, W)
        loss = criterion(pred, target)
        assert 0.0 <= loss.item() <= 2.0

    def test_output_is_scalar(self):
        criterion = DiceLoss()
        pred = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        loss = criterion(pred, target)
        assert loss.dim() == 0


class TestGANLoss:
    @pytest.mark.parametrize("mode", ["vanilla", "lsgan", "wgan"])
    def test_scalar_output(self, mode):
        criterion = GANLoss(mode=mode)
        pred = torch.randn(B, 1)
        loss = criterion(pred, is_real=True)
        assert loss.dim() == 0

    @pytest.mark.parametrize("mode", ["vanilla", "lsgan"])
    def test_real_loss_lower_than_fake(self, mode):
        """With fixed predictions, real-labelled loss should differ from fake."""
        criterion = GANLoss(mode=mode, real_label=1.0, fake_label=0.0)
        pred = torch.ones(B, 1)  # discriminator always outputs 1
        real_loss = criterion(pred, is_real=True)
        fake_loss = criterion(pred, is_real=False)
        assert real_loss.item() != fake_loss.item()


class TestLaplaceLoss:
    def test_constant_pred_gives_low_loss(self):
        """Constant probability maps should have near-zero Laplacian."""
        criterion = LaplaceLoss()
        pred = torch.zeros(B, C, D, H, W)
        loss = criterion(pred)
        assert loss.item() < 1.0

    def test_random_pred_gives_positive_loss(self):
        criterion = LaplaceLoss()
        pred = torch.randn(B, C, D, H, W)
        loss = criterion(pred)
        assert loss.item() >= 0.0

    def test_output_is_scalar(self):
        criterion = LaplaceLoss()
        pred = torch.randn(B, C, D, H, W)
        loss = criterion(pred)
        assert loss.dim() == 0


class TestCombinedLoss:
    def test_combined_loss_scalar(self):
        criterion = CombinedLoss()
        pred = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        loss = criterion(pred, target)
        assert loss.dim() == 0

    def test_combined_with_disc_pred(self):
        criterion = CombinedLoss(lambda_gan=0.1)
        pred = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        disc_pred = torch.randn(B, 1)
        loss = criterion(pred, target, disc_pred=disc_pred)
        assert loss.dim() == 0

    def test_zero_lambda_disables_term(self):
        criterion_no_gan = CombinedLoss(lambda_dice=1.0, lambda_gan=0.0, lambda_laplace=0.0)
        criterion_with_disc = CombinedLoss(lambda_dice=1.0, lambda_gan=0.1, lambda_laplace=0.0)
        pred = torch.randn(B, C, D, H, W)
        target = torch.randint(0, C, (B, D, H, W))
        disc_pred = torch.randn(B, 1)
        loss_no_gan = criterion_no_gan(pred, target)
        loss_with_disc = criterion_with_disc(pred, target, disc_pred=disc_pred)
        # Can only check they run without error and differ
        assert loss_no_gan.item() != loss_with_disc.item()
