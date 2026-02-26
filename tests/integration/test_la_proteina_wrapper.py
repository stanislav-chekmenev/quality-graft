"""Integration tests for LaProteinaWrapper.

Loads real checkpoints from ckpt/ and runs a small dummy batch through the
model on GPU.  Marked ``heavy`` so they are skipped in CI/CD unless
``--run-heavy`` is passed explicitly.

Run locally:
    pytest tests/integration/test_la_proteina_wrapper.py --run-heavy -v
"""

from pathlib import Path

import pytest
import torch

from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUNK_CKPT = PROJECT_ROOT / "ckpt" / "LD1_ucond_notri_512.ckpt"
AE_CKPT = PROJECT_ROOT / "ckpt" / "AE1_ucond_512.ckpt"

# ---------------------------------------------------------------------------
# Test geometry
# ---------------------------------------------------------------------------
BATCH_SIZE = 1
N_RESIDUES = 32
N_ATOMS = 37  # all-atom representation (atom37)

# ---------------------------------------------------------------------------
# Expected dimensions (from configs/model/la_proteina_wrapper.yaml)
# ---------------------------------------------------------------------------
TOKEN_DIM = 768
PAIR_DIM = 256
LATENT_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkpoints_available() -> bool:
    """Return True if both La-Proteina checkpoints exist on disk."""
    return TRUNK_CKPT.is_file() and AE_CKPT.is_file()

def _make_dummy_batch(
    batch_size: int = BATCH_SIZE,
    n_residues: int = N_RESIDUES,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a minimal batch that satisfies ``LaProteinaWrapper.forward()``.

    Tensor semantics
    ----------------
    - ``coords_nm``  : random all-atom coordinates in nanometres
    - ``coords``     : same structure, in Angstroms (used for torsion angles)
    - ``coord_mask`` : boolean atom mask (only backbone atoms N, CA, C, O set)
    - ``residue_type``: random amino-acid indices in [0, 19]
    - ``mask``       : all-True residue mask (no padding)
    - ``chains``     : single-chain (all zeros)
    """
    b, n, a = batch_size, n_residues, N_ATOMS

    # Random coordinates centred near zero, ~protein scale (nanometres)
    coords_nm = torch.randn(b, n, a, 3, device=device)
    # Angstrom version (some feature factories use this for torsion angles)
    coords = coords_nm * 10.0

    # Backbone atoms 0-3 (N, CA, C, O) are present; rest masked out
    coord_mask = torch.zeros(b, n, a, dtype=torch.bool, device=device)
    coord_mask[:, :, :4] = True

    residue_type = torch.randint(0, 20, (b, n), device=device)
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    chains = torch.zeros(b, n, dtype=torch.long, device=device)

    return {
        "coords_nm": coords_nm,
        "coords": coords,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "mask": mask,
        "chains": chains,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wrapper():
    """Load LaProteinaWrapper from real checkpoints (trunk-only, Option A).

    Module-scoped so the heavy checkpoint loading happens only once per
    test session.
    """
    model = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=str(TRUNK_CKPT),
        autoencoder_ckpt_path=str(AE_CKPT),
        use_decoder=False,
        t_value=1.0,
        deterministic_encode=True,
        device="cpu",
    )
    model = model.to("cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.heavy
@pytest.mark.skipif(not _checkpoints_available(), reason="Checkpoints not found in ckpt/")
class TestLaProteinaWrapperIntegration:
    """Integration tests that load real weights"""

    def test_forward_output_keys(self, wrapper):
        """Forward pass returns the expected output dictionary keys."""
        batch = _make_dummy_batch()
        outputs = wrapper(batch)

        expected_keys = {"trunk_seqs", "trunk_pair", "local_latents", "ca_coords"}
        assert set(outputs.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(outputs.keys())}"
        )

    def test_forward_output_shapes(self, wrapper):
        """All output tensors have the correct shapes."""
        batch = _make_dummy_batch()
        outputs = wrapper(batch)

        b, n = BATCH_SIZE, N_RESIDUES

        assert outputs["trunk_seqs"].shape == (b, n, TOKEN_DIM), (
            f"trunk_seqs: expected {(b, n, TOKEN_DIM)}, "
            f"got {outputs['trunk_seqs'].shape}"
        )
        assert outputs["trunk_pair"].shape == (b, n, n, PAIR_DIM), (
            f"trunk_pair: expected {(b, n, n, PAIR_DIM)}, "
            f"got {outputs['trunk_pair'].shape}"
        )
        assert outputs["local_latents"].shape == (b, n, LATENT_DIM), (
            f"local_latents: expected {(b, n, LATENT_DIM)}, "
            f"got {outputs['local_latents'].shape}"
        )
        assert outputs["ca_coords"].shape == (b, n, 3), (
            f"ca_coords: expected {(b, n, 3)}, "
            f"got {outputs['ca_coords'].shape}"
        )

    def test_forward_outputs_finite(self, wrapper):
        """No NaN or Inf values in any output tensor."""
        batch = _make_dummy_batch()
        outputs = wrapper(batch)

        for key, tensor in outputs.items():
            assert torch.isfinite(tensor).all(), (
                f"{key} contains non-finite values"
            )

    def test_no_gradients_on_wrapper(self, wrapper):
        """All wrapper parameters are frozen (require_grad=False)."""
        for name, param in wrapper.named_parameters():
            assert not param.requires_grad, (
                f"Parameter {name} has requires_grad=True but should be frozen"
            )

    def test_deterministic_two_passes(self, wrapper):
        """Two forward passes with the same input produce identical outputs.

        This validates ``deterministic_encode=True`` (posterior mean, no
        sampling) and the overall determinism of the frozen pipeline.
        """
        torch.manual_seed(42)
        batch = _make_dummy_batch()

        out1 = wrapper(batch)
        out2 = wrapper(batch)

        for key in out1:
            torch.testing.assert_close(
                out1[key], out2[key],
                msg=f"Non-deterministic output for '{key}'",
            )

    def test_masked_residues_zeroed(self, wrapper):
        """Outputs at masked (padding) positions should be zero.

        Masks the last residue and checks that per-residue outputs at that
        position are zeroed out by the mask application in the wrapper.
        """
        batch = _make_dummy_batch()
        # Mask out the last residue
        batch["mask"][:, -1] = False
        batch["coord_mask"][:, -1, :] = False

        outputs = wrapper(batch)

        # trunk_seqs and local_latents are multiplied by mask[..., None]
        assert (outputs["trunk_seqs"][:, -1, :] == 0).all(), (
            "trunk_seqs not zeroed at masked position"
        )
        assert (outputs["local_latents"][:, -1, :] == 0).all(), (
            "local_latents not zeroed at masked position"
        )
        assert (outputs["ca_coords"][:, -1, :] == 0).all(), (
            "ca_coords not zeroed at masked position"
        )
