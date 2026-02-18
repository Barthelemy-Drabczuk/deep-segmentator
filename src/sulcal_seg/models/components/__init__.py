"""Model component sub-packages (EAS, GAN, Laplace)."""
from .base_component import BaseComponent
# Import directly from engine modules (not via sub-package __init__)
# to avoid loading Pydantic-dependent config.py files at import time.
from .eas.eas_engine import EASEngine
from .gan.gan_engine import GANEngine
from .laplace.laplace_engine import LaplaceEngine

__all__ = ["BaseComponent", "EASEngine", "GANEngine", "LaplaceEngine"]
