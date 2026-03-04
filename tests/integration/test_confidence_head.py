"""Integration smoke tests for Boltz confidence head.

These tests verify that:
1) the confidence head initializes with plain dict config,
2) checkpoint weights load strictly with no missing/unexpected keys,
3) the simplified pLDDT-only forward pass (adaptor -> distogram -> pairformer
   -> linear heads) runs with just (s, z, mask).
"""

from __future__ import annotations

import warnings

from pathlib import Path

import pytest
import torch

from quality_graft.models.confidence_head import BoltzConfidenceHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_CKPT = PROJECT_ROOT / "ckpt" / "boltz1_conf.ckpt"

BATCH_SIZE = 1
N_TOKENS = 8

# Reference config values matching boltz1_conf.ckpt architecture
TOKEN_S = 384
TOKEN_Z = 128

CONFIDENCE_MODEL_ARGS = {
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
}

PAIRFORMER_ARGS = {
    "num_blocks": 48,
    "num_heads": 16,
    "dropout": 0.25,
    "post_layer_norm": False,
    "activation_checkpointing": False,
    "offload_to_cpu": False,
}

FULL_EMBEDDER_ARGS = {
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
}


def _checkpoint_available() -> bool:
    return CONF_CKPT.is_file()


def _make_head() -> BoltzConfidenceHead:
    """Create a BoltzConfidenceHead with the reference config."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        head = BoltzConfidenceHead(
            token_s=TOKEN_S,
            token_z=TOKEN_Z,
            pairformer_args=PAIRFORMER_ARGS,
            confidence_model_args=CONFIDENCE_MODEL_ARGS,
            full_embedder_args=FULL_EMBEDDER_ARGS,
            ckpt_path=str(CONF_CKPT),
            ckpt_prefix="confidence_module.",
            device="cuda",
        )

    assert len(caught) == 0, f"Unexpected warnings during load: {[str(w.message) for w in caught]}"
    return head


@pytest.mark.heavy
@pytest.mark.skipif(not _checkpoint_available(), reason="Confidence checkpoint not found in ckpt/")
class TestConfidenceHeadIntegration:
    """Heavy integration tests using real Boltz confidence checkpoint."""

    def test_init_and_strict_loading(self):
        """Plain dict init works and strict checkpoint loading succeeds."""
        head = _make_head()

        assert isinstance(head, BoltzConfidenceHead)
        assert head.confidence_module.imitate_trunk is True

    def test_custom_forward_smoke(self):
        """Simplified pLDDT-only forward runs with (s, z, mask)."""
        head = _make_head()
        head.eval()

        dev = next(head.parameters()).device
        b, n = BATCH_SIZE, N_TOKENS
        s = torch.randn(b, n, TOKEN_S, device=dev)
        z = torch.randn(b, n, n, TOKEN_Z, device=dev)
        mask = torch.ones(b, n, dtype=torch.float32, device=dev)

        with torch.no_grad():
            out = head(s=s, z=z, mask=mask)

        required = {"plddt_logits", "pde_logits", "resolved_logits"}
        assert required.issubset(out.keys())
        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
        assert out["resolved_logits"].shape == (b, n, 2)
