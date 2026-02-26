"""MONAI nnU-Net wrapper for sulcal segmentation (Phase 1).

Wraps the MONAI nnU-Net implementation as a drop-in model compatible
with the sulcal_seg training and validation pipelines.

Requirements (install before training):
    pip install "monai[nnunet,pytorch]"

Expected I/O:
    Input  : (B, 1, 256, 256, 256)  T1 MRI volume
    Output : (B, 52, 256, 256, 256) segmentation logits
    Classes: 52 sulcal labels (51 sulci + background)

Target performance (Phase 1 gate):
    Validation Dice ≥ 95%
    GPU inference   ~ 45 sec / subject
    Training time   ~ 3-4 days on single A100
    Model size      ~ 30M parameters
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from monai.losses import DiceCELoss
    from monai.networks.nets import DynUNet

    HAS_MONAI = True
except ImportError:  # pragma: no cover
    HAS_MONAI = False


# nnU-Net-style kernel sizes and strides for 256³ input
# 5 pooling stages: 256→128→64→32→16→8
_NNUNET_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_NNUNET_KERNELS = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]


class MONAInnUNetModel(nn.Module):
    """
    Self-configuring nnU-Net for 52-class sulcal segmentation.

    Uses MONAI's DynUNet as the backbone — the closest equivalent to the
    nnU-Net architecture available in MONAI without a full nnU-Net
    experiment configuration.  Replace with ``monai.apps.nnunet`` when
    running on a cluster with the full nnU-Net self-configuration pipeline.

    Args:
        config: Component config dict.  Recognised keys:
            input_channels  (int, default 1)
            output_channels (int, default 52)
            deep_supervision (bool, default True) — nnU-Net default.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if not HAS_MONAI:
            raise ImportError(
                "MONAI is required for MONAInnUNetModel. "
                "Install with: pip install 'monai[nnunet,pytorch]'"
            )

        cfg = config or {}
        in_ch: int = cfg.get("input_channels", 1)
        out_ch: int = cfg.get("output_channels", 52)
        deep_sup: bool = cfg.get("deep_supervision", True)

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=_NNUNET_KERNELS,
            strides=_NNUNET_STRIDES,
            upsample_kernel_size=_NNUNET_STRIDES[1:],
            deep_supervision=deep_sup,
            deep_supr_num=3,
            res_block=True,
        )

        # DiceCELoss: combines soft Dice + cross-entropy (nnU-Net default)
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: T1 MRI volume (B, 1, 256, 256, 256).

        Returns:
            Segmentation logits (B, 52, 256, 256, 256).
            When deep_supervision=True, training returns a list of
            multi-scale outputs; use the first element for inference.
        """
        return self.model(x)

    def compute_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DiceCE loss, handling deep supervision outputs.

        Args:
            output: Model output — (B, 52, D, H, W) or list of tensors
                    when deep_supervision=True.
            target: Integer label map (B, 1, D, H, W).

        Returns:
            Scalar loss tensor.
        """
        if isinstance(output, (list, tuple)):
            # Deep supervision: average losses across scales (weights
            # could be tuned; equal weighting is a reasonable default).
            return sum(self.loss_fn(o, target) for o in output) / len(output)
        return self.loss_fn(output, target)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities for inference."""
        logits = self.forward(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]  # highest-resolution output
        return torch.softmax(logits, dim=1)
