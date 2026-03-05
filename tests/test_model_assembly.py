"""Tests for the full QualityGraft model assembly.

Unit tests (no ``heavy`` marker)
    Use mock / lightweight stand-ins for the frozen sub-modules so they run
    in CI without real checkpoints.

Heavy integration smoke tests (``heavy`` marker)
    Load real La-Proteina + Boltz1 confidence checkpoints, assemble the
    full ``QualityGraft`` model, and run a forward pass on CPU.  Requires
    ``--run-heavy``.

Run:
    pytest tests/test_model_assembly.py -v                   # unit tests only
    pytest tests/test_model_assembly.py -v --run-heavy       # unit + heavy
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.quality_graft import QualityGraft


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUNK_CKPT = PROJECT_ROOT / "ckpt" / "LD1_ucond_notri_512.ckpt"
AE_CKPT = PROJECT_ROOT / "ckpt" / "AE1_ucond_512.ckpt"
CONF_CKPT = PROJECT_ROOT / "ckpt" / "boltz1_conf.ckpt"

# La-Proteina dimensions
TRUNK_DIM = 768
PAIR_DIM = 256
LATENT_DIM = 8

# Boltz1 target dimensions
TARGET_S_DIM = 384
TARGET_Z_DIM = 128

# Test geometry
BATCH_SIZE = 2
N_RESIDUES = 10


# ---------------------------------------------------------------------------
# Helpers: lightweight mocks for unit tests
# ---------------------------------------------------------------------------


class _MockLaProteinaWrapper(nn.Module):
    """Minimal mock that returns random tensors of the right shape."""

    def __init__(self, use_decoder: bool = False):
        super().__init__()
        self.use_decoder = use_decoder
        # A dummy parameter so the module registers on the model
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        b = batch["mask"].shape[0]
        n = batch["mask"].shape[1]
        device = batch["mask"].device
        dtype = torch.float32
        out = {
            "trunk_seqs": torch.randn(b, n, TRUNK_DIM, device=device, dtype=dtype),
            "trunk_pair": torch.randn(b, n, n, PAIR_DIM, device=device, dtype=dtype),
            "local_latents": torch.randn(b, n, LATENT_DIM, device=device, dtype=dtype),
            "ca_coords": torch.randn(b, n, 3, device=device, dtype=dtype),
        }
        if self.use_decoder:
            out["decoder_seqs"] = torch.randn(
                b, n, TRUNK_DIM, device=device, dtype=dtype
            )
        return out


class _MockConfidenceHead(nn.Module):
    """Minimal mock that returns logits derived from inputs (gradient-preserving).

    Uses simple linear projections so gradients flow from the output back
    through ``s`` and ``z`` to the adaptor.
    """

    def __init__(self):
        super().__init__()
        # A dummy parameter so the module registers on the model
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        # Simple projections that preserve the gradient graph
        self._s_to_plddt = nn.Linear(TARGET_S_DIM, 50, bias=False)
        self._s_to_resolved = nn.Linear(TARGET_S_DIM, 2, bias=False)
        self._z_to_pde = nn.Linear(TARGET_Z_DIM, 64, bias=False)
        # Freeze (these are part of the "frozen" confidence head mock)
        self._s_to_plddt.requires_grad_(False)
        self._s_to_resolved.requires_grad_(False)
        self._z_to_pde.requires_grad_(False)

    def forward(self, s, z, mask, use_kernels=False):
        return {
            "plddt_logits": self._s_to_plddt(s),              # [b, n, 50]
            "pde_logits": self._z_to_pde(z + z.transpose(1, 2)),  # [b, n, n, 64]
            "resolved_logits": self._s_to_resolved(s),         # [b, n, 2]
        }


def _make_mock_model(
    source_mode: str = "trunk",
    n_attn_layers: int = 1,
) -> QualityGraft:
    """Build a QualityGraft with mock frozen sub-modules."""
    la_proteina = _MockLaProteinaWrapper(
        use_decoder=(source_mode == "hybrid"),
    )
    adaptor = AdaptorModule(
        source_mode=source_mode,
        trunk_dim=TRUNK_DIM,
        pair_dim=PAIR_DIM,
        latent_dim=LATENT_DIM,
        target_s_dim=TARGET_S_DIM,
        target_z_dim=TARGET_Z_DIM,
        n_attn_layers=n_attn_layers,
    )
    confidence_head = _MockConfidenceHead()

    return QualityGraft(
        la_proteina=la_proteina,
        adaptor=adaptor,
        confidence_head=confidence_head,
    )


def _make_dummy_batch(
    batch_size: int = BATCH_SIZE,
    n_residues: int = N_RESIDUES,
) -> dict[str, torch.Tensor]:
    """Build a minimal batch for the mock La-Proteina wrapper.

    The mock wrapper only reads ``batch["mask"]``, but we provide all
    standard fields for forward-compatibility.
    """
    b, n = batch_size, n_residues
    return {
        "coords_nm": torch.randn(b, n, 37, 3),
        "coord_mask": torch.ones(b, n, 37, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (b, n)),
        "mask": torch.ones(b, n, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Unit tests (no heavy marker, no real checkpoints)
# ---------------------------------------------------------------------------


class TestQualityGraftUnit:
    """Unit tests for QualityGraft with mock sub-modules."""

    def test_forward_output_keys(self):
        """Forward pass returns expected output keys."""
        model = _make_mock_model()
        batch = _make_dummy_batch()
        out = model(batch)

        expected_keys = {"plddt_logits", "pde_logits", "resolved_logits"}
        assert set(out.keys()) == expected_keys

    def test_forward_output_shapes(self):
        """Output tensors have correct shapes."""
        model = _make_mock_model()
        batch = _make_dummy_batch()
        out = model(batch)

        b, n = BATCH_SIZE, N_RESIDUES
        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
        assert out["resolved_logits"].shape == (b, n, 2)

    def test_trainable_parameters_only_adaptor(self):
        """Only adaptor parameters should be trainable."""
        model = _make_mock_model()

        trainable = model.trainable_parameters()
        trainable_names = {
            name for name, p in model.named_parameters() if p.requires_grad
        }

        # All trainable params should be in the adaptor sub-module
        for name in trainable_names:
            assert name.startswith("adaptor."), (
                f"Non-adaptor parameter is trainable: {name}"
            )

        # Adaptor should have trainable parameters
        assert len(trainable) > 0

    def test_num_trainable_parameters(self):
        """num_trainable_parameters matches adaptor parameter count."""
        model = _make_mock_model(n_attn_layers=0)

        adaptor_params = sum(
            p.numel() for p in model.adaptor.parameters()
        )
        assert model.num_trainable_parameters() == adaptor_params

    def test_num_frozen_parameters(self):
        """num_frozen_parameters counts La-Proteina + confidence head params."""
        model = _make_mock_model()

        frozen = model.num_frozen_parameters()
        # Mock La-Proteina: 1 dummy param
        # Mock Confidence Head: 1 dummy + 3 frozen linear projections
        expected_frozen = (
            1  # _MockLaProteinaWrapper._dummy
            + 1  # _MockConfidenceHead._dummy
            + TARGET_S_DIM * 50  # _s_to_plddt
            + TARGET_S_DIM * 2  # _s_to_resolved
            + TARGET_Z_DIM * 64  # _z_to_pde
        )
        assert frozen == expected_frozen, (
            f"Expected {expected_frozen} frozen params, got {frozen}"
        )

    def test_hybrid_mode_forward(self):
        """Hybrid mode model produces correct output shapes."""
        model = _make_mock_model(source_mode="hybrid")
        batch = _make_dummy_batch()
        out = model(batch)

        b, n = BATCH_SIZE, N_RESIDUES
        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
        assert out["resolved_logits"].shape == (b, n, 2)

    def test_gradient_flows_through_adaptor(self):
        """Gradients from loss flow back through the adaptor."""
        model = _make_mock_model(n_attn_layers=1)
        batch = _make_dummy_batch()
        out = model(batch)

        # Simulate a combined loss that uses both s-path (pLDDT) and z-path (PDE)
        # to ensure gradients flow through all adaptor parameters
        loss = out["plddt_logits"].sum() + out["pde_logits"].sum()
        loss.backward()

        for name, param in model.adaptor.named_parameters():
            assert param.grad is not None, f"No gradient for adaptor.{name}"

    def test_linear_only_adaptor(self):
        """n_attn_layers=0 produces valid output."""
        model = _make_mock_model(n_attn_layers=0)
        batch = _make_dummy_batch()
        out = model(batch)

        assert out["plddt_logits"].shape == (BATCH_SIZE, N_RESIDUES, 50)

    def test_different_sequence_lengths(self):
        """Model handles varying sequence lengths."""
        model = _make_mock_model()

        for n in [5, 16, 32]:
            batch = _make_dummy_batch(batch_size=1, n_residues=n)
            out = model(batch)
            assert out["plddt_logits"].shape == (1, n, 50), (
                f"Failed for n={n}"
            )

    def test_single_residue(self):
        """Edge case: single-residue protein."""
        model = _make_mock_model(n_attn_layers=0)
        batch = _make_dummy_batch(batch_size=1, n_residues=1)
        out = model(batch)

        assert out["plddt_logits"].shape == (1, 1, 50)
        assert out["pde_logits"].shape == (1, 1, 1, 64)
        assert out["resolved_logits"].shape == (1, 1, 2)


# ---------------------------------------------------------------------------
# Heavy integration tests (require --run-heavy + real checkpoints)
# ---------------------------------------------------------------------------


def _all_checkpoints_available() -> bool:
    """Check that all required checkpoints exist."""
    return TRUNK_CKPT.is_file() and AE_CKPT.is_file() and CONF_CKPT.is_file()


@pytest.fixture(scope="module")
def full_model():
    """Assemble a real QualityGraft model from checkpoints on CPU.

    Module-scoped so the expensive checkpoint loading happens only once.
    Uses a small adaptor (n_attn_layers=0) to keep the test fast.
    """
    from quality_graft.models.confidence_head import BoltzConfidenceHead
    from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper

    # Load La-Proteina wrapper
    la_proteina = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=str(TRUNK_CKPT),
        autoencoder_ckpt_path=str(AE_CKPT),
        use_decoder=False,
        t_value=1.0,
        deterministic_encode=True,
        device="cpu",
    )

    # Create adaptor (trainable, small for speed)
    adaptor = AdaptorModule(
        source_mode="trunk",
        trunk_dim=TRUNK_DIM,
        pair_dim=PAIR_DIM,
        latent_dim=LATENT_DIM,
        target_s_dim=TARGET_S_DIM,
        target_z_dim=TARGET_Z_DIM,
        n_attn_layers=0,
    )

    # Load confidence head
    confidence_head = BoltzConfidenceHead(
        token_s=TARGET_S_DIM,
        token_z=TARGET_Z_DIM,
        pairformer_args={
            "num_blocks": 48,
            "num_heads": 16,
            "dropout": 0.25,
            "post_layer_norm": False,
            "activation_checkpointing": False,
            "offload_to_cpu": False,
        },
        confidence_model_args={
            "num_dist_bins": 64,
            "max_dist": 22,
            "add_s_to_z_prod": True,
            "add_s_input_to_s": True,
            "use_s_diffusion": True,
            "add_z_input_to_z": True,
            "confidence_args": {
                "num_plddt_bins": 50,
                "num_pde_bins": 64,
                "num_pae_bins": 64,
            },
        },
        full_embedder_args={
            "atom_s": 128,
            "atom_z": 16,
            "token_s": 384,
            "token_z": 128,
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
            "atom_feature_dim": 389,
            "no_atom_encoder": False,
            "atom_encoder_depth": 3,
            "atom_encoder_heads": 4,
        },
        ckpt_path=str(CONF_CKPT),
        ckpt_prefix="confidence_module.",
        device="cpu",
        freeze=True,
        strict_loading=True,
    )

    model = QualityGraft(
        la_proteina=la_proteina,
        adaptor=adaptor,
        confidence_head=confidence_head,
    )
    model.eval()
    return model


def _make_real_batch(
    batch_size: int = 1,
    n_residues: int = 8,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a batch compatible with the real La-Proteina wrapper."""
    b, n, a = batch_size, n_residues, 37

    coords_nm = torch.randn(b, n, a, 3, device=device)
    coords = coords_nm * 10.0
    coord_mask = torch.zeros(b, n, a, dtype=torch.bool, device=device)
    coord_mask[:, :, :4] = True  # backbone atoms N, CA, C, O

    return {
        "coords_nm": coords_nm,
        "coords": coords,
        "coord_mask": coord_mask,
        "residue_type": torch.randint(0, 20, (b, n), device=device),
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "chains": torch.zeros(b, n, dtype=torch.long, device=device),
    }


@pytest.mark.heavy
@pytest.mark.skipif(
    not _all_checkpoints_available(),
    reason="One or more checkpoints not found in ckpt/",
)
class TestQualityGraftHeavy:
    """Heavy integration smoke tests with real checkpoints on CPU."""

    def test_full_forward_pass(self, full_model):
        """End-to-end forward: raw batch → quality predictions."""
        batch = _make_real_batch(batch_size=1, n_residues=8)

        with torch.no_grad():
            out = full_model(batch)

        assert "plddt_logits" in out
        assert "pde_logits" in out
        assert "resolved_logits" in out

        assert out["plddt_logits"].shape == (1, 8, 50)
        assert out["pde_logits"].shape == (1, 8, 8, 64)
        assert out["resolved_logits"].shape == (1, 8, 2)

    def test_outputs_are_finite(self, full_model):
        """No NaN or Inf values in output tensors."""
        batch = _make_real_batch(batch_size=1, n_residues=8)

        with torch.no_grad():
            out = full_model(batch)

        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), (
                f"{key} contains non-finite values"
            )

    def test_only_adaptor_trainable(self, full_model):
        """Only adaptor parameters should have requires_grad=True."""
        trainable_names = [
            name for name, p in full_model.named_parameters() if p.requires_grad
        ]
        frozen_names = [
            name for name, p in full_model.named_parameters() if not p.requires_grad
        ]

        # All trainable params should be in adaptor
        for name in trainable_names:
            assert name.startswith("adaptor."), (
                f"Non-adaptor parameter is trainable: {name}"
            )

        # La-Proteina and confidence head should be frozen
        assert any(n.startswith("la_proteina.") for n in frozen_names)
        assert any(n.startswith("confidence_head.") for n in frozen_names)

    def test_gradient_flows_to_adaptor(self, full_model):
        """Gradients from a dummy loss reach all adaptor parameters."""
        batch = _make_real_batch(batch_size=1, n_residues=8)

        # Enable grad for this test (model is in eval mode from fixture)
        out = full_model(batch)
        loss = out["plddt_logits"].sum()
        loss.backward()

        for name, param in full_model.adaptor.named_parameters():
            assert param.grad is not None, (
                f"No gradient for adaptor.{name}"
            )

        # Clean up
        full_model.zero_grad()

    def test_deterministic_two_passes(self, full_model):
        """Two forward passes with the same input produce identical results."""
        torch.manual_seed(42)
        batch = _make_real_batch(batch_size=1, n_residues=8)

        with torch.no_grad():
            out1 = full_model(batch)
            out2 = full_model(batch)

        for key in out1:
            torch.testing.assert_close(
                out1[key], out2[key],
                msg=f"Non-deterministic output for '{key}'",
            )

    def test_parameter_count_sanity(self, full_model):
        """Total trainable params should be in the adaptor-only range."""
        n_trainable = full_model.num_trainable_parameters()
        n_frozen = full_model.num_frozen_parameters()

        # Adaptor with n_attn_layers=0: ~332K params (linear projections only)
        assert 300_000 < n_trainable < 500_000, (
            f"Unexpected trainable param count: {n_trainable}"
        )

        # Confidence head alone is ~152.7M; La-Proteina is also large
        assert n_frozen > 100_000_000, (
            f"Unexpected frozen param count: {n_frozen}"
        )
