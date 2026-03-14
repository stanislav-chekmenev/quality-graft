"""PyTorch Lightning module for Quality-Graft training.

Wraps QualityGraft(nn.Module) with training/validation steps,
masked pLDDT cross-entropy loss, and validation metrics.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from loguru import logger
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
        debug_mode: bool = False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.num_plddt_bins = num_plddt_bins
        self.debug_mode = debug_mode
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    # Diagnostic: count modules in train vs eval per component
    # ------------------------------------------------------------------
    def _count_modes(self, module: torch.nn.Module) -> tuple[int, int]:
        """Return (n_train, n_eval) for all sub-modules."""
        n_train = n_eval = 0
        for m in module.modules():
            if m.training:
                n_train += 1
            else:
                n_eval += 1
        return n_train, n_eval

    def _log_mode_summary(self, phase: str, step: int) -> None:
        """Log train/eval mode counts for each component."""
        components = {
            "la_proteina": self.model.la_proteina,
            "adaptor": self.model.adaptor,
            "confidence_head": self.model.confidence_head,
        }
        parts = [f"[{phase} step={step}]"]
        for name, mod in components.items():
            n_train, n_eval = self._count_modes(mod)
            parts.append(f"{name}: train={n_train} eval={n_eval}")

        # Check a pairformer layer directly
        pf = self.model.confidence_head.confidence_module.pairformer_module
        pf_layer0_training = pf.layers[0].training if len(pf.layers) > 0 else None
        parts.append(f"pairformer_layer0.training={pf_layer0_training}")

        logger.info(" | ".join(parts))

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
            ignore_index=-1,
        )
        loss = loss.view_as(plddt_labels) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def on_train_epoch_start(self):
        """Keep frozen components in eval() after Lightning's model.train()."""
        self.model.la_proteina.eval()
        self.model.confidence_head.eval()
        self.model.adaptor.train()
        if self.debug_mode:
            val_freq = self.trainer.check_val_every_n_epoch or 1
            if self.current_epoch % val_freq == 0 or self.current_epoch < 3:
                self._log_mode_summary("on_train_epoch_start", self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        """Re-enforce eval() on frozen components — check both independently."""
        needs_fix = False
        if self.model.la_proteina.training:
            self.model.la_proteina.eval()
            needs_fix = True
        if self.model.confidence_head.training:
            self.model.confidence_head.eval()
            needs_fix = True
        if needs_fix:
            logger.warning(
                f"[on_train_batch_start epoch={self.current_epoch} batch={batch_idx}] "
                "Had to re-enforce eval() on frozen components!"
            )
            if self.debug_mode:
                self._log_mode_summary("after_fix", self.global_step)

    def on_validation_epoch_start(self):
        """Put entire model in eval() for validation."""
        self.model.eval()
        if self.debug_mode:
            self._log_mode_summary("on_validation_epoch_start", self.current_epoch)

    def on_validation_epoch_end(self):
        """Log modes right after validation ends (before Lightning restores train)."""
        if self.debug_mode:
            self._log_mode_summary("on_validation_epoch_end", self.current_epoch)

    def training_step(self, batch, batch_idx):
        if self.debug_mode:
            val_freq = self.trainer.check_val_every_n_epoch or 1
            if self.current_epoch % val_freq == 0 or self.current_epoch < 3:
                self._log_mode_summary("training_step_BEFORE_forward", self.global_step)
        outputs = self.model(batch)
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.float()
        loss = self._compute_loss(outputs["plddt_logits"], batch["plddt_bin"], mask)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._log_mode_summary("validation_step_BEFORE_forward", self.global_step)
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