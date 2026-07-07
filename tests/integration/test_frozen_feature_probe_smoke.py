"""Heavy smoke test for the frozen-feature probe end-to-end.

Loads the real La-Proteina checkpoints, runs a wrapper forward on a
dummy batch, builds all four probe variants, and fits each for a handful
of steps — asserting finite Pearson and correct feature dimensionality.
Marked ``heavy`` (skipped unless ``--run-heavy``) and skipped entirely if
the checkpoints are absent.

Run locally:
    pytest tests/integration/test_frozen_feature_probe_smoke.py --run-heavy -v
"""

from pathlib import Path

import math

import pytest
import torch

from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper
from quality_graft.probes.frozen_feature_probe import (
    PLDDT_VARIANTS,
    ProbeData,
    ProbePartition,
    build_probe_features,
    fit_probe,
)
from quality_graft.data.plddt_utils import NUM_PLDDT_BINS, plddt_to_bin

# Reuse the checkpoint/dummy-batch helpers from the wrapper integration test.
from tests.integration.test_la_proteina_wrapper import (
    _checkpoints_available,
    _make_dummy_batch,
    TRUNK_CKPT,
    AE_CKPT,
)


_EXPECTED_D_IN = {"s_only": 768, "s_latents": 776, "z_pooled": 768, "s_z": 1536}


@pytest.mark.heavy
@pytest.mark.skipif(
    not _checkpoints_available(), reason="Checkpoints not found in ckpt/"
)
def test_wrapper_features_to_fit_runs_and_finite() -> None:
    wrapper = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=str(TRUNK_CKPT),
        autoencoder_ckpt_path=str(AE_CKPT),
        use_decoder=False,
        t_value=0.99,
        deterministic_encode=True,
        device="cpu",
    )
    wrapper.eval()
    assert abs(wrapper.t_value - 0.99) < 1e-9

    batch = _make_dummy_batch(batch_size=2, n_residues=24)
    with torch.no_grad():
        reprs = wrapper(batch)
    mask = batch["mask"]

    # Fabricate pLDDT bin labels aligned to the residue axis (no real
    # labels in the dummy batch); the probe only needs finite, in-range
    # bins to exercise the CE fit and the metric reductions.
    torch.manual_seed(0)
    plddt = torch.rand(mask.shape)
    plddt_bin = plddt_to_bin(plddt)

    inter = {
        "trunk_seqs": reprs["trunk_seqs"],
        "trunk_pair": reprs["trunk_pair"],
        "local_latents": reprs["local_latents"],
        "mask": mask,
        "plddt_bin": plddt_bin,
        "plddt_mask": mask,
    }

    for variant in PLDDT_VARIANTS:
        X, y_bin, mask_eff = build_probe_features(inter, variant=variant)
        assert X.shape[-1] == _EXPECTED_D_IN[variant]
        assert X.dtype == torch.float32
        assert X.shape[1] == y_bin.shape[1] == mask_eff.shape[1]

        part = ProbePartition(
            X=X, y_bin=y_bin.long(), mask=mask_eff, num_bins=NUM_PLDDT_BINS
        )
        data = ProbeData(train_fit=part, train_eval=part, val_eval=part)
        trace = fit_probe(data, steps=5, lr=1e-3, warmup_steps=0, seed=0)

        final = trace[-1]
        for key in ("train_fit_pearson", "train_eval_pearson", "val_eval_pearson"):
            assert math.isfinite(final[key]), f"{variant}: {key} not finite"
