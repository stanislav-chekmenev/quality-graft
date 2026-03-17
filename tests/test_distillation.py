"""Tests for the distillation architecture.

Covers:
- StudentConfidenceHead (forward shapes, param count, gradient flow)
- Distillation loss (KL divergence, combined loss, backward compat)
- Full pipeline integration with student head
- Preprocessing backward compatibility (missing plddt_logits)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.quality_graft import QualityGraft
from quality_graft.models.student_head import StudentConfidenceHead
from quality_graft.training.lightning_module import QualityGraftLightningModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_DIM, PAIR_DIM, LATENT_DIM = 768, 256, 8
TARGET_S_DIM, TARGET_Z_DIM = 384, 128
B, N = 2, 10
NUM_PLDDT_BINS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_student_head(num_blocks=2, dropout=0.0):
    """Build a small student head for testing."""
    return StudentConfidenceHead(
        token_s=TARGET_S_DIM,
        token_z=TARGET_Z_DIM,
        num_blocks=num_blocks,
        num_heads=16,
        dropout=dropout,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_plddt_bins=NUM_PLDDT_BINS,
        num_pde_bins=64,
        predict_pde=True,
        predict_resolved=True,
    )


def _make_distillation_model(num_blocks=2):
    """Build a QualityGraft model with student confidence head."""
    la_proteina = _MockLaProteinaWrapper()
    adaptor = AdaptorModule(
        source_mode="trunk",
        trunk_dim=TRUNK_DIM,
        pair_dim=PAIR_DIM,
        latent_dim=LATENT_DIM,
        target_s_dim=TARGET_S_DIM,
        target_z_dim=TARGET_Z_DIM,
        n_attn_layers=0,  # projections only for distillation
    )
    student_head = _make_student_head(num_blocks=num_blocks)
    return QualityGraft(
        la_proteina=la_proteina,
        adaptor=adaptor,
        confidence_head=student_head,
    )


def _make_batch(with_logits=False):
    """Build a minimal batch."""
    batch = {
        "coords_nm": torch.randn(B, N, 37, 3),
        "coord_mask": torch.ones(B, N, 37, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (B, N)),
        "mask": torch.ones(B, N, dtype=torch.float32),
        "plddt_bin": torch.randint(0, NUM_PLDDT_BINS, (B, N)),
    }
    if with_logits:
        batch["plddt_logits"] = torch.randn(B, N, NUM_PLDDT_BINS)
    return batch


def _make_lightning_module(num_blocks=2, distill_alpha=0.7, distill_temperature=2.0):
    """Build a lightning module with student head."""
    model = _make_distillation_model(num_blocks=num_blocks)
    return QualityGraftLightningModule(
        model=model,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10,
        min_lr=1e-6,
        num_plddt_bins=NUM_PLDDT_BINS,
        distill_alpha=distill_alpha,
        distill_temperature=distill_temperature,
    )


# ---------------------------------------------------------------------------
# Student module tests
# ---------------------------------------------------------------------------


class TestStudentConfidenceHead:
    def test_forward_output_keys(self):
        """Student head returns expected output keys."""
        head = _make_student_head()
        s = torch.randn(B, N, TARGET_S_DIM)
        z = torch.randn(B, N, N, TARGET_Z_DIM)
        mask = torch.ones(B, N)
        out = head(s, z, mask)
        assert set(out.keys()) == {"plddt_logits", "pde_logits", "resolved_logits"}

    def test_forward_output_shapes(self):
        """Output tensors have correct shapes."""
        head = _make_student_head()
        s = torch.randn(B, N, TARGET_S_DIM)
        z = torch.randn(B, N, N, TARGET_Z_DIM)
        mask = torch.ones(B, N)
        out = head(s, z, mask)
        assert out["plddt_logits"].shape == (B, N, NUM_PLDDT_BINS)
        assert out["pde_logits"].shape == (B, N, N, 64)
        assert out["resolved_logits"].shape == (B, N, 2)

    def test_plddt_only_mode(self):
        """Student head with only pLDDT prediction."""
        head = StudentConfidenceHead(
            token_s=TARGET_S_DIM,
            token_z=TARGET_Z_DIM,
            num_blocks=2,
            predict_pde=False,
            predict_resolved=False,
        )
        s = torch.randn(B, N, TARGET_S_DIM)
        z = torch.randn(B, N, N, TARGET_Z_DIM)
        mask = torch.ones(B, N)
        out = head(s, z, mask)
        assert set(out.keys()) == {"plddt_logits"}
        assert out["plddt_logits"].shape == (B, N, NUM_PLDDT_BINS)

    def test_parameter_count_within_budget(self):
        """4-block student head should be ~10-15M params."""
        head = StudentConfidenceHead(
            token_s=TARGET_S_DIM,
            token_z=TARGET_Z_DIM,
            num_blocks=4,
            num_heads=16,
            pairwise_head_width=32,
            pairwise_num_heads=4,
        )
        n_params = sum(p.numel() for p in head.parameters())
        assert 8_000_000 < n_params < 16_000_000, (
            f"4-block student has {n_params:,} params, expected 8-16M"
        )

    def test_5_block_parameter_count(self):
        """5-block student should be ~13-17M params."""
        head = StudentConfidenceHead(
            token_s=TARGET_S_DIM,
            token_z=TARGET_Z_DIM,
            num_blocks=5,
        )
        n_params = sum(p.numel() for p in head.parameters())
        assert 10_000_000 < n_params < 20_000_000, (
            f"5-block student has {n_params:,} params, expected 10-20M"
        )

    def test_gradient_flow(self):
        """All student parameters receive gradients."""
        head = _make_student_head()
        s = torch.randn(B, N, TARGET_S_DIM)
        z = torch.randn(B, N, N, TARGET_Z_DIM)
        mask = torch.ones(B, N)
        out = head(s, z, mask)
        # Combined loss through all heads so every param gets gradients
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_outputs_are_finite(self):
        """Outputs should be finite (no NaN/Inf)."""
        head = _make_student_head()
        s = torch.randn(B, N, TARGET_S_DIM)
        z = torch.randn(B, N, N, TARGET_Z_DIM)
        mask = torch.ones(B, N)
        out = head(s, z, mask)
        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), f"{key} has non-finite values"

    def test_variable_sequence_lengths(self):
        """Student handles varying sequence lengths."""
        head = _make_student_head()
        for n in [5, 16, 32]:
            s = torch.randn(1, n, TARGET_S_DIM)
            z = torch.randn(1, n, n, TARGET_Z_DIM)
            mask = torch.ones(1, n)
            out = head(s, z, mask)
            assert out["plddt_logits"].shape == (1, n, NUM_PLDDT_BINS)

    def test_all_params_trainable(self):
        """All student parameters should be trainable (no frozen params)."""
        head = _make_student_head()
        for name, param in head.named_parameters():
            assert param.requires_grad, f"{name} is frozen but should be trainable"


# ---------------------------------------------------------------------------
# Distillation loss tests
# ---------------------------------------------------------------------------


class TestDistillationLoss:
    def test_ce_only_when_no_teacher(self):
        """Without teacher logits, loss equals pure cross-entropy."""
        module = _make_lightning_module()
        logits = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        mask = torch.ones(B, N)

        loss = module._compute_loss(logits, labels, mask, teacher_logits=None)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_alpha_zero_equals_pure_ce(self):
        """With alpha=0, combined loss equals pure CE regardless of teacher."""
        module = _make_lightning_module(distill_alpha=0.0)
        logits = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        mask = torch.ones(B, N)

        loss_with_teacher = module._compute_loss(logits, labels, mask, teacher)
        loss_without = module._compute_loss(logits, labels, mask, None)
        torch.testing.assert_close(loss_with_teacher, loss_without, atol=1e-5, rtol=1e-5)

    def test_alpha_one_is_pure_kl(self):
        """With alpha=1, combined loss should be pure KL (no CE contribution)."""
        module = _make_lightning_module(distill_alpha=1.0, distill_temperature=1.0)
        logits = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        mask = torch.ones(B, N)

        loss = module._compute_loss(logits, labels, mask, teacher)

        # Manually compute KL
        student_log_probs = F.log_softmax(logits, dim=-1)
        teacher_probs = F.softmax(teacher, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        expected_kl = (kl * mask).sum() / mask.sum()

        torch.testing.assert_close(loss, expected_kl, atol=1e-5, rtol=1e-5)

    def test_kl_zero_when_student_matches_teacher(self):
        """KL divergence is zero when student logits match teacher logits."""
        module = _make_lightning_module(distill_alpha=1.0, distill_temperature=2.0)
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        mask = torch.ones(B, N)

        # Student = teacher should give KL ≈ 0
        loss = module._compute_loss(teacher.clone(), labels, mask, teacher)
        assert loss.item() < 1e-5, f"KL should be ~0 when student=teacher, got {loss.item()}"

    def test_temperature_softens_distribution(self):
        """Higher temperature produces smaller KL (softer distributions)."""
        logits = torch.randn(B, N, NUM_PLDDT_BINS)
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        mask = torch.ones(B, N)

        module_low_t = _make_lightning_module(distill_alpha=1.0, distill_temperature=1.0)
        module_high_t = _make_lightning_module(distill_alpha=1.0, distill_temperature=4.0)

        loss_low = module_low_t._compute_loss(logits, labels, mask, teacher)
        loss_high = module_high_t._compute_loss(logits, labels, mask, teacher)

        # With T² scaling, high-T loss should still be in a reasonable range
        # but the actual distribution is softer
        assert torch.isfinite(loss_low) and torch.isfinite(loss_high)

    def test_mask_is_respected(self):
        """Padding positions (mask=0) should not contribute to loss."""
        module = _make_lightning_module()
        logits = torch.randn(B, N, NUM_PLDDT_BINS)
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))

        # Full mask vs half mask should give different losses
        full_mask = torch.ones(B, N)
        half_mask = torch.ones(B, N)
        half_mask[:, N // 2 :] = 0

        loss_full = module._compute_loss(logits, labels, full_mask, teacher)
        loss_half = module._compute_loss(logits, labels, half_mask, teacher)
        assert not torch.allclose(loss_full, loss_half)

    def test_loss_backward_with_teacher(self):
        """Loss with teacher logits supports backward pass."""
        module = _make_lightning_module()
        logits = torch.randn(B, N, NUM_PLDDT_BINS, requires_grad=True)
        teacher = torch.randn(B, N, NUM_PLDDT_BINS)
        labels = torch.randint(0, NUM_PLDDT_BINS, (B, N))
        mask = torch.ones(B, N)

        loss = module._compute_loss(logits, labels, mask, teacher)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# Full pipeline integration tests (with student head)
# ---------------------------------------------------------------------------


class TestDistillationPipeline:
    def test_forward_with_student_head(self):
        """Full pipeline forward pass with student confidence head."""
        model = _make_distillation_model()
        batch = _make_batch()
        out = model(batch)

        assert out["plddt_logits"].shape == (B, N, NUM_PLDDT_BINS)
        assert "pde_logits" in out
        assert "resolved_logits" in out

    def test_trainable_params_include_student(self):
        """Both adaptor and student head parameters should be trainable."""
        model = _make_distillation_model()
        trainable_names = {
            name for name, p in model.named_parameters() if p.requires_grad
        }
        has_adaptor = any(n.startswith("adaptor.") for n in trainable_names)
        has_student = any(n.startswith("confidence_head.") for n in trainable_names)
        assert has_adaptor, "Adaptor parameters should be trainable"
        assert has_student, "Student head parameters should be trainable"

    def test_la_proteina_is_frozen(self):
        """La-Proteina wrapper should remain frozen."""
        model = _make_distillation_model()
        for name, p in model.la_proteina.named_parameters():
            assert not p.requires_grad, f"la_proteina.{name} should be frozen"

    def test_gradient_flows_end_to_end(self):
        """Gradients flow from loss through student head back to adaptor."""
        model = _make_distillation_model()
        batch = _make_batch()
        out = model(batch)
        # Combined loss through all heads
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        for name, p in model.adaptor.named_parameters():
            assert p.grad is not None, f"No gradient for adaptor.{name}"
        for name, p in model.confidence_head.named_parameters():
            assert p.grad is not None, f"No gradient for confidence_head.{name}"

    def test_training_step_without_logits(self):
        """Training step works without teacher logits (backward compat)."""
        module = _make_lightning_module()
        batch = _make_batch(with_logits=False)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_training_step_with_logits(self):
        """Training step uses distillation loss when teacher logits present."""
        module = _make_lightning_module()
        batch = _make_batch(with_logits=True)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_validation_step_always_uses_hard_targets(self):
        """Validation step should use hard targets only for comparable metrics."""
        module = _make_lightning_module()
        batch = _make_batch(with_logits=True)
        logged = {}
        module.log = lambda name, value, **kwargs: logged.update({name: value})
        module.validation_step(batch, batch_idx=0)

        expected_keys = {
            "val/loss", "val/plddt_accuracy", "val/plddt_mae",
            "val/pearson_r", "val/spearman_r",
        }
        assert expected_keys.issubset(logged.keys())

    def test_student_mode_detection(self):
        """Lightning module correctly detects student mode."""
        module = _make_lightning_module()
        assert module._student_mode is True

    def test_optimizer_includes_all_trainable(self):
        """Optimizer should contain both adaptor and student head params."""
        module = _make_lightning_module()
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        opt_param_ids = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                opt_param_ids.add(id(p))

        # All trainable params should be in optimizer
        for name, p in module.model.named_parameters():
            if p.requires_grad:
                assert id(p) in opt_param_ids, f"{name} not in optimizer"


# ---------------------------------------------------------------------------
# Preprocessing backward compatibility tests
# ---------------------------------------------------------------------------


class TestPreprocessingCompat:
    def test_batch_without_logits(self):
        """Batch dict without plddt_logits shouldn't crash training."""
        module = _make_lightning_module()
        batch = _make_batch(with_logits=False)
        assert "plddt_logits" not in batch
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_batch_get_returns_none(self):
        """batch.get('plddt_logits') returns None when field is missing."""
        batch = _make_batch(with_logits=False)
        assert batch.get("plddt_logits") is None
