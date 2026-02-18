"""Learning rate schedulers for sulcal segmentation training."""
import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine annealing.

    During warm-up (steps 0 → warmup_steps):
        lr = base_lr * (step / warmup_steps)

    After warm-up (steps warmup_steps → T_max):
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * progress))

    where progress = (step - warmup_steps) / (T_max - warmup_steps).

    Args:
        optimizer: Wrapped optimiser.
        T_max: Total number of training steps (not including warm-up).
        warmup_steps: Number of linear warm-up steps.
        eta_min: Minimum learning rate after annealing.
        last_epoch: The index of the last epoch (default -1 = start from beginning).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warm-up
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Cosine annealing
        progress = (step - self.warmup_steps) / max(1, self.T_max - self.warmup_steps)
        progress = min(1.0, progress)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]
