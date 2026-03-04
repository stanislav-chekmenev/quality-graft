"""Integration smoke tests for Boltz confidence head.

These tests verify that:
1) the confidence head initializes from Hydra/OmegaConf config,
2) checkpoint weights load strictly with no missing/unexpected keys,
3) the simplified pLDDT-only forward pass (adaptor -> distogram -> pairformer
   -> linear heads) runs with just (s, z, ca_coords, mask).
"""

from __future__ import annotations

import warnings

from pathlib import Path

import hydra
import pytest
import torch

from quality_graft.models.confidence_head import BoltzConfidenceHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_CKPT = PROJECT_ROOT / "ckpt" / "boltz1_conf.ckpt"

BATCH_SIZE = 1
N_TOKENS = 8


def _checkpoint_available() -> bool:
    return CONF_CKPT.is_file()


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
        """Simplified pLDDT-only forward runs with (s, z, mask)."""
        head = self._instantiate_from_hydra()
        head.eval()

        b, n = BATCH_SIZE, N_TOKENS
        s = torch.randn(b, n, 384)
        z = torch.randn(b, n, n, 128)
        mask = torch.ones(b, n, dtype=torch.float32)

        with torch.no_grad():
            out = head(s=s, z=z, mask=mask)

        required = {"plddt_logits", "pde_logits", "resolved_logits"}
        assert required.issubset(out.keys())
        assert out["plddt_logits"].shape == (b, n, 50)
        assert out["pde_logits"].shape == (b, n, n, 64)
        assert out["resolved_logits"].shape == (b, n, 2)

