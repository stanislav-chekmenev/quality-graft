"""PyTorch Lightning module for Quality-Graft training.

Wraps QualityGraft(nn.Module) with training/validation steps,
masked pLDDT cross-entropy loss, and validation metrics.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.metrics import (
    plddt_accuracy,
    plddt_mae,
    pearson_r,
    spearman_r,
    _logits_to_continuous,
    _labels_to_continuous,
)


class QualityGraftLightningModule(L.LightningModule):
    """Lightning wrapper for the Quality-Graft model.

    Parameters
    ----------
    model : QualityGraft
        The assembled model (La-Proteina + adaptor + confidence head).
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    betas : tuple[float, float]
        AdamW betas.
    warmup_steps : int
        Number of linear warmup steps.
    min_lr : float
        Minimum learning rate after linear decay.
    num_plddt_bins : int
        Number of pLDDT bins (default 50).
    """

    def __init__(
        self,
        model: QualityGraft,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 1e-6,
        num_plddt_bins: int = 50,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.num_plddt_bins = num_plddt_bins
        self.save_hyperparameters(ignore=["model"])

    def _compute_loss(self, plddt_logits: Tensor, plddt_labels: Tensor, mask: Tensor) -> Tensor:
        """Masked cross-entropy loss over pLDDT bins.

        Parameters
        ----------
        plddt_logits : [b, n, num_bins]
        plddt_labels : [b, n] long
        mask : [b, n] float (1=valid, 0=padding)
        """
        loss = F.cross_entropy(
            plddt_logits.reshape(-1, self.num_plddt_bins),
            plddt_labels.reshape(-1),
            reduction="none",
        )
        loss = loss.view_as(plddt_labels) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.float()
        loss = self._compute_loss(outputs["plddt_logits"], batch["plddt_bin"], mask)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.float()
        logits = outputs["plddt_logits"]
        labels = batch["plddt_bin"]

        # Loss
        loss = self._compute_loss(logits, labels, mask)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Metrics
        acc = plddt_accuracy(logits, labels, mask)
        mae = plddt_mae(logits, labels, mask, self.num_plddt_bins)

        pred_cont = _logits_to_continuous(logits, self.num_plddt_bins)
        target_cont = _labels_to_continuous(labels, self.num_plddt_bins)
        pr = pearson_r(pred_cont, target_cont, mask)
        sr = spearman_r(pred_cont, target_cont, mask)

        self.log("val/plddt_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log("val/plddt_mae", mae, sync_dist=True)
        self.log("val/pearson_r", pr, sync_dist=True)
        self.log("val/spearman_r", sr, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._lr_lambda
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _lr_lambda(self, step: int) -> float:
        """Linear warmup then linear decay to min_lr."""
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        # Linear decay from 1.0 to min_lr/lr over remaining training
        total_steps = self.trainer.estimated_stepping_batches
        decay_steps = total_steps - self.warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - self.warmup_steps) / decay_steps
        min_factor = self.min_lr / self.lr
        return max(1.0 - progress * (1.0 - min_factor), min_factor)