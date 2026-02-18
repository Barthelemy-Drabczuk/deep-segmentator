"""GAN segmentation component."""
# Config import deferred to avoid Pydantic version conflicts at import time.
# Import directly: from sulcal_seg.models.components.gan.config import GANComponentConfig
from .discriminator import Discriminator
from .dual_label_trainer import DualLabelTrainer
from .gan_engine import GANEngine
from .generator import ConvBlock3D, Generator
from .losses import CombinedLoss, DiceLoss, GANLoss, LaplaceLoss
from .mode_collapse_detector import ModeCollapseDetector, compute_entropy

__all__ = [
    "GANEngine",
    "Generator",
    "ConvBlock3D",
    "Discriminator",
    "DualLabelTrainer",
    "DiceLoss",
    "GANLoss",
    "LaplaceLoss",
    "CombinedLoss",
    "ModeCollapseDetector",
    "compute_entropy",
]
