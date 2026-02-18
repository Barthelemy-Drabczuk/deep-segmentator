"""Model architecture configurations."""
from pydantic import BaseModel, Field


class GeneratorConfig(BaseModel):
    """Configuration for the GAN generator (3D U-Net)."""

    encoder_depth: int = 4
    base_filters: int = 32
    filter_multiplier: int = 2
    num_sulci_labels: int = 40
    use_attention: bool = False
    conv_type: str = "conv3d"
    activation: str = "leaky_relu"


class DiscriminatorConfig(BaseModel):
    """Configuration for the GAN discriminator."""

    base_filters: int = 32
    num_layers: int = 4
    use_spectral_norm: bool = True
    num_sulci_labels: int = 40
