"""MONAI nnU-Net trainer for Phase 1 sulcal segmentation.

Handles the training loop, validation, checkpointing, and TensorBoard
logging for ``MONAInnUNetModel``.  GPU + real data loaders are required
to run the actual training; the core methods raise ``NotImplementedError``
stubs when called without a configured environment.

Usage:
    trainer = MonaiTrainer(config, model, train_loader, val_loader)
    trainer.train()          # runs for config["num_epochs"] epochs
    trainer.load_best()      # loads best checkpoint
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sulcal_seg.models.monai_nnunet import MONAInnUNetModel
from sulcal_seg.utils.checkpoint_manager import CheckpointManager


class MonaiTrainer:
    """
    Training orchestrator for ``MONAInnUNetModel``.

    Args:
        config: Training configuration dict.  Recognised keys:

            num_epochs        (int,   default 80)
            learning_rate     (float, default 1e-3)
            weight_decay      (float, default 1e-5)
            grad_clip_norm    (float, default 1.0)
            val_interval      (int,   default 1)   — validate every N epochs
            checkpoint_dir    (str,   default "outputs/checkpoints")
            log_dir           (str,   default "outputs/logs")
            mixed_precision   (bool,  default True)

        model: Initialised ``MONAInnUNetModel``.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        device: torch.device (defaults to CUDA if available).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: MONAInnUNetModel,
        train_loader: Any,
        val_loader: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.num_epochs: int = config.get("num_epochs", 80)
        self.lr: float = config.get("learning_rate", 1e-3)
        self.grad_clip: float = config.get("grad_clip_norm", 1.0)
        self.val_interval: int = config.get("val_interval", 1)
        self.amp: bool = config.get("mixed_precision", True)

        ckpt_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_manager = CheckpointManager(ckpt_dir)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.best_val_dice: float = 0.0
        self.best_epoch: int = 0
        self._writer: Any = None  # TensorBoard SummaryWriter, lazily created

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop for ``num_epochs`` epochs.

        Returns:
            Dict with ``'best_val_dice'`` and ``'best_epoch'``.

        Raises:
            NotImplementedError: Always — requires GPU and real DataLoaders.
                                  Implement by replacing the raise below.
        """
        # TODO: Uncomment and run on a machine with GPU + real data:
        #
        # self.model.to(self.device)
        # self._init_writer()
        # for epoch in range(1, self.num_epochs + 1):
        #     train_loss = self._train_epoch(epoch)
        #     if epoch % self.val_interval == 0:
        #         val_dice = self._validate(epoch)
        #         ckpt_state = {
        #             "epoch": epoch,
        #             "model_state_dict": self.model.state_dict(),
        #             "optimizer_state_dict": self.optimizer.state_dict(),
        #             "metrics": {"val_dice": val_dice},
        #         }
        #         self.checkpoint_manager.save(
        #             ckpt_state,
        #             f"checkpoint_epoch_{epoch:04d}.pt",
        #         )
        #         if val_dice > self.best_val_dice:
        #             self.best_val_dice = val_dice
        #             self.best_epoch = epoch
        #             self.checkpoint_manager.save(ckpt_state, "best_model.pt")
        #     self._log_scalars(epoch, train_loss=train_loss, val_dice=val_dice)
        # self._writer.close()
        # return {"best_val_dice": self.best_val_dice, "best_epoch": self.best_epoch}
        raise NotImplementedError(
            "MonaiTrainer.train() requires a GPU and real DataLoaders. "
            "Uncomment the training loop above and run on appropriate hardware."
        )

    def load_best(self) -> int:
        """
        Load the best checkpoint saved during training.

        Returns:
            The epoch number of the loaded checkpoint.
        """
        path = self.checkpoint_manager.get_best() or self.checkpoint_manager.get_latest()
        if path is None:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoint_manager.checkpoint_dir}"
            )
        state = self.checkpoint_manager.load(path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        return state.get("epoch", 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp):
                output = self.model(images)
                loss = self.model.compute_loss(output, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, epoch: int) -> float:
        """
        Run validation and return mean Dice score over all classes.

        TODO: Replace the stub below with MONAI sliding-window inference
              for full 256³ volumes (use ``monai.inferers.SlidingWindowInferer``).
        """
        raise NotImplementedError(
            "MonaiTrainer._validate() requires GPU + real validation data. "
            "Use monai.inferers.SlidingWindowInferer for full-volume inference."
        )

    def _log_scalars(self, epoch: int, **kwargs: float) -> None:
        """Write scalars to TensorBoard."""
        if self._writer is None:
            return
        for name, value in kwargs.items():
            self._writer.add_scalar(name, value, epoch)

    def _init_writer(self) -> None:
        """Lazily initialise TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self.config.get("log_dir", "outputs/logs")
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print(
                "TensorBoard not available. "
                "Install with: pip install tensorboard"
            )
