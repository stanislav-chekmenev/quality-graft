"""Tests for QualityGraftLightningModule.

Uses mock sub-modules (same as test_model_assembly.py) to avoid checkpoints.
"""

import pytest
import torch
import torch.nn as nn

from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.lightning_module import QualityGraftLightningModule

# Dimensions
TRUNK_DIM, PAIR_DIM, LATENT_DIM = 768, 256, 8
TARGET_S_DIM, TARGET_Z_DIM = 384, 128
B, N = 2, 10


class _MockLaProteinaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        b, n = batch["mask"].shape
        device = batch["mask"].device
        return {
            "trunk_seqs": torch.randn(b, n, TRUNK_DIM, device=device),
            "trunk_pair": torch.randn(b, n, n, PAIR_DIM, device=device),
            "local_latents": torch.randn(b, n, LATENT_DIM, device=device),
            "ca_coords": torch.randn(b, n, 3, device=device),
        }


class _MockConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self._s_to_plddt = nn.Linear(TARGET_S_DIM, 50, bias=False)
        self._s_to_resolved = nn.Linear(TARGET_S_DIM, 2, bias=False)
        self._z_to_pde = nn.Linear(TARGET_Z_DIM, 64, bias=False)
        self.requires_grad_(False)

    def forward(self, s, z, mask, use_kernels=False):
        return {
            "plddt_logits": self._s_to_plddt(s),
            "pde_logits": self._z_to_pde(z + z.transpose(1, 2)),
            "resolved_logits": self._s_to_resolved(s),
        }


def _make_module(n_attn_layers=0):
    model = QualityGraft(
        la_proteina=_MockLaProteinaWrapper(),
        adaptor=AdaptorModule(
            source_mode="trunk",
            trunk_dim=TRUNK_DIM, pair_dim=PAIR_DIM, latent_dim=LATENT_DIM,
            target_s_dim=TARGET_S_DIM, target_z_dim=TARGET_Z_DIM,
            n_attn_layers=n_attn_layers,
        ),
        confidence_head=_MockConfidenceHead(),
    )
    return QualityGraftLightningModule(
        model=model,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10,
        min_lr=1e-6,
        num_plddt_bins=50,
    )


def _make_batch():
    return {
        "coords_nm": torch.randn(B, N, 37, 3),
        "coord_mask": torch.ones(B, N, 37, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (B, N)),
        "mask": torch.ones(B, N, dtype=torch.float32),
        "plddt_bin": torch.randint(0, 50, (B, N)),
    }


class TestTrainingStep:
    def test_returns_scalar_loss(self):
        module = _make_module()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_is_finite(self):
        module = _make_module()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)


class TestValidationStep:
    def test_logs_all_metrics(self):
        module = _make_module()
        batch = _make_batch()
        # Capture logged metrics
        logged = {}
        module.log = lambda name, value, **kwargs: logged.update({name: value})
        module.validation_step(batch, batch_idx=0)
        expected_keys = {"val/loss", "val/plddt_accuracy", "val/plddt_mae",
                         "val/pearson_r", "val/spearman_r"}
        assert expected_keys.issubset(logged.keys())


class TestConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self):
        module = _make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_only_adaptor_params_in_optimizer(self):
        module = _make_module()
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        opt_params = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                opt_params.add(id(p))
        adaptor_params = {id(p) for p in module.model.adaptor.parameters()}
        assert opt_params == adaptor_params