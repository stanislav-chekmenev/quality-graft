"""Unit tests for the QG frozen-feature linear-probe diagnostic.

Ports the Complexa probe unit suite to quality-graft. PAE variants are
dropped (SwissProt has only pLDDT), a ``s_latents`` variant is added
(the honest adaptor-input floor: ``trunk_seqs`` + ``local_latents``),
the bin-center tensor is replaced by ``num_bins: int`` (QG metrics take
an int, not a centers tensor), and the split reproduces the QG random
``split_dataframe`` val set (seed 42) rather than a cluster-disjoint
Teddymer split.

Fast, no checkpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quality_graft.probes.frozen_feature_probe import (
    LinearProbe,
    ProbeData,
    ProbePartition,
    build_probe_features,
    fit_probe,
    pool_pair_features,
    probe_metrics,
    split_swissprot_for_probe,
)
from quality_graft.training.metrics import (
    _labels_to_continuous,
    _logits_to_continuous,
    pearson_r,
    spearman_r,
)


# ---------------------------------------------------------------------------
# 1. pool_pair_features — masked mean/max over valid j, diag, finite
# ---------------------------------------------------------------------------

def test_pool_pair_features_masked_mean_max_diag() -> None:
    # b=1, n=3, d_pair=2; residue 2 is padded, so only j in {0, 1} are valid.
    z = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[5.0, 6.0], [7.0, 8.0], [200.0, 200.0]],
                [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])

    out = pool_pair_features(z, mask)
    assert out.shape == (1, 3, 6)

    mean_part = out[..., :2]
    max_part = out[..., 2:4]
    diag_part = out[..., 4:6]

    # mean over valid j in {0, 1} only; padded residue i=2 is a zeroed row.
    expected_mean = torch.tensor([[[2.0, 3.0], [6.0, 7.0], [0.0, 0.0]]])
    torch.testing.assert_close(mean_part, expected_mean)

    # max over valid j in {0, 1} only — padded j=2 (100/200) excluded;
    # padded residue i=2 zeroed (not -inf).
    expected_max = torch.tensor([[[3.0, 4.0], [7.0, 8.0], [0.0, 0.0]]])
    torch.testing.assert_close(max_part, expected_max)

    # diag is z[:, k, k, :] verbatim.
    expected_diag = torch.tensor([[[1.0, 2.0], [7.0, 8.0], [9.0, 9.0]]])
    torch.testing.assert_close(diag_part, expected_diag)

    assert torch.isfinite(out).all()


def test_pool_pair_features_empty_row_zeroed() -> None:
    # Row 0 has one valid j (itself); row 1 has NO valid j at all.
    z = torch.tensor([[[[-5.0], [1000.0]], [[1000.0], [-7.0]]]])
    mask = torch.tensor([[True, False]])

    out = pool_pair_features(z, mask)
    assert out.shape == (1, 2, 3)

    max_part = out[..., 1:2]
    torch.testing.assert_close(max_part[0, 0], torch.tensor([-5.0]))
    torch.testing.assert_close(max_part[0, 1], torch.tensor([0.0]))

    mean_part = out[..., 0:1]
    torch.testing.assert_close(mean_part[0, 0], torch.tensor([-5.0]))
    torch.testing.assert_close(mean_part[0, 1], torch.tensor([0.0]))

    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2. build_probe_features — variant dims, fp32, residue-axis alignment
# ---------------------------------------------------------------------------

B, N, TOKEN_DIM, PAIR_DIM, LATENT_DIM = 2, 5, 768, 256, 8
N_PLDDT_BINS = 50


def _inter(n_label: int = N) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(B, N, dtype=torch.bool)
    return {
        "trunk_seqs": torch.randn(B, N, TOKEN_DIM, generator=g),
        "trunk_pair": torch.randn(B, N, N, PAIR_DIM, generator=g),
        "local_latents": torch.randn(B, N, LATENT_DIM, generator=g),
        "mask": mask,
        "plddt_bin": torch.randint(0, N_PLDDT_BINS, (B, n_label), generator=g),
        "plddt_mask": torch.ones(B, n_label, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "variant,d_in",
    [("s_only", 768), ("s_latents", 776), ("z_pooled", 768), ("s_z", 1536)],
)
def test_build_probe_features_variant_dims_and_alignment(
    variant: str, d_in: int
) -> None:
    inter = _inter()
    X, y_bin, mask_eff = build_probe_features(inter, variant=variant)

    assert X.shape == (B, N, d_in)
    assert X.dtype == torch.float32
    assert X.shape[1] == N == y_bin.shape[1] == mask_eff.shape[1]


def test_build_probe_features_raises_on_axis_mismatch() -> None:
    inter = _inter(n_label=N + 1)
    with pytest.raises((ValueError, AssertionError)):
        build_probe_features(inter, variant="s_only")


def test_build_probe_features_unknown_variant_raises() -> None:
    inter = _inter()
    with pytest.raises(ValueError):
        build_probe_features(inter, variant="pae_z_direct")


# ---------------------------------------------------------------------------
# 3. only the LinearProbe trains
# ---------------------------------------------------------------------------

BB, NN, DD, NBINS = 2, 6, 8, 5


def _partition(seed: int) -> ProbePartition:
    g = torch.Generator().manual_seed(seed)
    return ProbePartition(
        X=torch.randn(BB, NN, DD, generator=g),
        y_bin=torch.randint(0, NBINS, (BB, NN), generator=g),
        mask=torch.ones(BB, NN, dtype=torch.bool),
        num_bins=NBINS,
    )


def _data() -> ProbeData:
    return ProbeData(
        train_fit=_partition(0),
        train_eval=_partition(1),
        val_eval=_partition(2),
    )


def test_only_linear_probe_trains(monkeypatch) -> None:
    data = _data()

    captured_params: list[list[torch.nn.Parameter]] = []
    captured_probes: list[LinearProbe] = []

    real_linear_init = LinearProbe.__init__

    def spy_init(self, d_in: int, n_out: int) -> None:
        real_linear_init(self, d_in, n_out)
        captured_probes.append(self)

    monkeypatch.setattr(LinearProbe, "__init__", spy_init)

    for opt_name in ("Adam", "AdamW", "SGD"):
        real_opt = getattr(torch.optim, opt_name)

        def make_spy(real):
            def spy(params, *args, **kwargs):
                params = list(params)
                captured_params.append(params)
                return real(params, *args, **kwargs)

            return spy

        monkeypatch.setattr(torch.optim, opt_name, make_spy(real_opt))

    fit_probe(data, steps=1, lr=1e-2, warmup_steps=0, seed=0)

    assert captured_probes, "fit_probe never constructed a LinearProbe"
    probe = captured_probes[-1]
    probe_param_ids = {id(p) for p in probe.parameters()}

    assert captured_params, "fit_probe never constructed an optimizer"
    for params in captured_params:
        assert {id(p) for p in params} == probe_param_ids, (
            "optimizer received parameters other than LinearProbe.parameters()"
        )

    assert probe.linear.weight.grad is not None
    assert probe.linear.bias.grad is not None

    for part in (data.train_fit, data.train_eval, data.val_eval):
        assert part.X.requires_grad is False
        assert part.X.is_leaf
        assert part.X.grad is None


# ---------------------------------------------------------------------------
# 4. fit_probe deterministic under seed
# ---------------------------------------------------------------------------

def test_fit_probe_deterministic_under_seed() -> None:
    trace_a = fit_probe(
        _data(), steps=5, lr=1e-2, warmup_steps=2, seed=123, log_every=1
    )
    trace_b = fit_probe(
        _data(), steps=5, lr=1e-2, warmup_steps=2, seed=123, log_every=1
    )

    assert len(trace_a) == len(trace_b)
    for ra, rb in zip(trace_a, trace_b):
        assert ra.keys() == rb.keys()
        for k in ra:
            assert ra[k] == rb[k], f"trace diverged at key {k!r}"


# ---------------------------------------------------------------------------
# 5. probe_metrics reuses the per-protein QG metrics (num_bins int)
# ---------------------------------------------------------------------------

MET_BINS = 8


def _pooled_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.reshape(-1)
    t = target.reshape(-1)
    p_c = p - p.mean()
    t_c = t - t.mean()
    return (p_c * t_c).sum() / (p_c.pow(2).sum() * t_c.pow(2).sum()).sqrt()


def test_probe_metrics_reuses_per_protein_metrics() -> None:
    g = torch.Generator().manual_seed(0)
    b, n = 2, 12

    logits = torch.randn(b, n, MET_BINS, generator=g)
    y_bin = torch.stack(
        [
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
            torch.tensor([7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2]),
        ]
    )
    mask = torch.ones(b, n, dtype=torch.bool)

    pred_cont = _logits_to_continuous(logits, MET_BINS)
    target_cont = _labels_to_continuous(y_bin, MET_BINS)

    ref_pearson = pearson_r(pred_cont, target_cont, mask)
    ref_spearman = spearman_r(pred_cont, target_cont, mask)
    pooled = _pooled_pearson(pred_cont, target_cont)

    assert not torch.isclose(ref_pearson, pooled, atol=1e-3), (
        "fixture degenerate: per-protein and pooled Pearson coincide"
    )

    out = probe_metrics(logits, y_bin, mask, MET_BINS)
    assert torch.isclose(torch.tensor(out["pearson"]), ref_pearson, atol=1e-6)
    assert torch.isclose(torch.tensor(out["spearman"]), ref_spearman, atol=1e-6)


def test_probe_metrics_invariant_to_padded_label_flips() -> None:
    g = torch.Generator().manual_seed(0)
    b, n, n_valid = 2, 8, 5
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :n_valid] = True

    logits = torch.randn(b, n, MET_BINS, generator=g)
    y_bin = torch.randint(0, MET_BINS, (b, n), generator=g)

    y_flipped = y_bin.clone()
    pad = ~mask
    y_flipped[pad] = (y_flipped[pad] + 3) % MET_BINS

    out_a = probe_metrics(logits, y_bin, mask, MET_BINS)
    out_b = probe_metrics(logits, y_flipped, mask, MET_BINS)

    assert out_a["pearson"] == out_b["pearson"]
    assert out_a["spearman"] == out_b["spearman"]


# ---------------------------------------------------------------------------
# 6. split reproduces the QG val set (split_dataframe seed=42)
# ---------------------------------------------------------------------------

def _fixture_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"P{i:04d}" for i in range(n)]})


def test_split_reproduces_qg_val_set() -> None:
    from la_proteina.proteinfoundation.utils.cluster_utils import split_dataframe

    df = _fixture_df()
    train_fit, train_eval, val_eval = split_swissprot_for_probe(
        df, train_val_test=(0.94, 0.03, 0.03), split_seed=42, probe_seed=0
    )

    ref = split_dataframe(df, ["train", "val", "test"], [0.94, 0.03, 0.03], seed=42)

    assert set(val_eval["id"]) == set(ref["val"]["id"])

    probe_train_ids = set(train_fit["id"]) | set(train_eval["id"])
    assert probe_train_ids == set(ref["train"]["id"])

    # All three partitions row-disjoint.
    assert set(train_fit["id"]).isdisjoint(set(train_eval["id"]))
    assert probe_train_ids.isdisjoint(set(val_eval["id"]))

    # Non-empty.
    assert len(train_fit) > 0 and len(train_eval) > 0 and len(val_eval) > 0


def test_split_probe_seed_reproducible() -> None:
    df = _fixture_df()
    a = split_swissprot_for_probe(df, probe_seed=0)
    b = split_swissprot_for_probe(df, probe_seed=0)
    for pa, pb in zip(a, b):
        assert set(pa["id"]) == set(pb["id"])
