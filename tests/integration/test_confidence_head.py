"""Integration smoke tests for Boltz confidence head.

These tests verify that:
1) the confidence head initializes from Hydra/OmegaConf config,
2) checkpoint weights load strictly with no missing/unexpected keys,
3) the custom forward pass (adaptor -> distogram -> pairformer -> heads) runs.
"""

from __future__ import annotations

import warnings

from pathlib import Path

import hydra
import pytest
import torch

from boltz.data import const
from quality_graft.models.confidence_head import BoltzConfidenceHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_CKPT = PROJECT_ROOT / "ckpt" / "boltz1_conf.ckpt"

BATCH_SIZE = 1
N_TOKENS = 8


def _checkpoint_available() -> bool:
    return CONF_CKPT.is_file()


def _make_smoke_features(batch_size: int, n_tokens: int) -> dict[str, torch.Tensor]:
    """Create a minimal feature dictionary required by confidence forward."""
    b, n = batch_size, n_tokens
    n_atoms = n

    token_to_rep_atom = torch.eye(n, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1)
    atom_to_token = torch.eye(n_atoms, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1)

    token_pad_mask = torch.ones(b, n, dtype=torch.bool)
    atom_pad_mask = torch.ones(b, n_atoms, dtype=torch.bool)

    asym_id = torch.zeros(b, n, dtype=torch.long)
    mol_type = torch.full(
        (b, n),
        fill_value=const.chain_type_ids["PROTEIN"],
        dtype=torch.long,
    )

    idx = torch.arange(n, dtype=torch.long)
    frames_idx = torch.stack([idx, idx, idx], dim=-1).unsqueeze(0).repeat(b, 1, 1)

    return {
        "token_to_rep_atom": token_to_rep_atom,
        "token_pad_mask": token_pad_mask,
        "atom_to_token": atom_to_token,
        "atom_pad_mask": atom_pad_mask,
        "asym_id": asym_id,
        "mol_type": mol_type,
        "frames_idx": frames_idx,
    }


def _make_single_sequence_msa_features(
    batch_size: int,
    n_tokens: int,
    aa_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build minimal MSA features for a single random amino-acid sequence."""
    b, n = batch_size, n_tokens
    m = 1

    msa = torch.nn.functional.one_hot(
        aa_token_ids.long(),
        num_classes=const.num_tokens,
    ).float().unsqueeze(1)

    has_deletion = torch.zeros(b, m, n, dtype=torch.float32)
    deletion_value = torch.zeros(b, m, n, dtype=torch.float32)
    msa_paired = torch.zeros(b, m, n, dtype=torch.float32)
    msa_mask = torch.ones(b, m, n, dtype=torch.float32)
    token_pad_mask = torch.ones(b, n, dtype=torch.bool)

    return {
        "msa": msa,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": msa_paired,
        "msa_mask": msa_mask,
        "token_pad_mask": token_pad_mask,
    }


@pytest.mark.heavy
@pytest.mark.skipif(not _checkpoint_available(), reason="Confidence checkpoint not found in ckpt/")
class TestConfidenceHeadIntegration:
    """Heavy integration tests using real Boltz confidence checkpoint."""

    def _instantiate_from_hydra(self) -> BoltzConfidenceHead:
        config_dir = str(PROJECT_ROOT / "configs")
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=hydra.__version__):
            conf_cfg = hydra.compose(config_name="model/confidence_head").model

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            head = BoltzConfidenceHead.from_hydra_config(conf_cfg)

        assert len(caught) == 0, f"Unexpected warnings during load: {[str(w.message) for w in caught]}"
        return head

    def test_hydra_init_and_strict_loading(self):
        """Hydra/OmegaConf init works and strict checkpoint loading succeeds."""
        head = self._instantiate_from_hydra()

        assert isinstance(head, BoltzConfidenceHead)
        assert head.confidence_module.imitate_trunk is True

    def test_custom_forward_smoke(self):
        """Custom confidence forward runs and returns key outputs with expected shapes."""
        head = self._instantiate_from_hydra()
        head.eval()

        b, n = BATCH_SIZE, N_TOKENS
        s = torch.randn(b, n, 384)
        z = torch.randn(b, n, n, 128)
        x_pred = torch.randn(b, n, 3)
        pred_distogram_logits = torch.randn(b, n, n, 64)
        feats = _make_smoke_features(batch_size=b, n_tokens=n)

        with torch.no_grad():
            out = head(
                s=s,
                z=z,
                x_pred=x_pred,
                feats=feats,
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=1,
                s_diffusion=torch.randn(b, n, 2 * 384),
            )

        required = {"plddt_logits", "pde_logits", "resolved_logits", "pae_logits"}
        assert required.issubset(out.keys())
        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
        assert out["resolved_logits"].shape == (b, n, 2)
        assert out["pae_logits"].shape == (b, n, n, 64)

    def test_custom_forward_with_single_seq_msa_smoke(self):
        """Custom confidence forward runs with an optional single-sequence MSA."""
        head = self._instantiate_from_hydra()
        head.eval()

        b, n = BATCH_SIZE, N_TOKENS
        s = torch.randn(b, n, 384)
        z = torch.randn(b, n, n, 128)
        x_pred = torch.randn(b, n, 3)
        pred_distogram_logits = torch.randn(b, n, n, 64)
        feats = _make_smoke_features(batch_size=b, n_tokens=n)

        aa_token_ids = torch.randint(2, 22, (b, n))
        msa_feats = _make_single_sequence_msa_features(
            batch_size=b,
            n_tokens=n,
            aa_token_ids=aa_token_ids,
        )

        with torch.no_grad():
            out = head(
                s=s,
                z=z,
                x_pred=x_pred,
                feats=feats,
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=1,
                s_diffusion=torch.randn(b, n, 2 * 384),
                msa_feats=msa_feats,
                msa_aa_tokens=aa_token_ids,
            )

        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
