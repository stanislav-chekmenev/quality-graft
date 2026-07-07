"""Frozen-feature linear-probe diagnostic for the La-Proteina trunk.

Fits a single ``nn.Linear`` on the frozen wrapper's features
(``trunk_seqs``, ``local_latents``, ``trunk_pair``) to predict AF2
B-factor pLDDT bin labels. The probe answers a single diagnostic
question: are the frozen features near-linearly label-predictive (which
would explain a distillation run's fast convergence) rather than the
fast convergence being a val-split leak? Only the ``LinearProbe`` trains;
the frozen trunk is run under ``torch.no_grad()`` upstream by the
harness.

Ported from the Complexa diagnostic. Differences:

- The bin-center tensor is replaced by ``num_bins: int`` because
  ``quality_graft.training.metrics`` takes a bin count, not a centers
  tensor.
- Feature source is the ``LaProteinaWrapper`` output rather than a
  Complexa trunk hook. There is no extended-frame trimming
  (``n_orig``): QG batches carry no extended frame, so features and
  labels are already on the same residue axis. Alignment is asserted
  (hard ``ValueError`` on mismatch).
- Variants are pLDDT-only (SwissProt has no PAE) and add ``s_latents``
  (``trunk_seqs`` + ``local_latents``), the honest adaptor-input floor.
- ``split_swissprot_for_probe`` reproduces the QG *random*
  ``split_dataframe`` val set (seed 42) and carves the train side into a
  fit/eval pair with a seeded random split. This carve is NOT
  cluster-disjoint (SwissProt has no cluster column), so family leak
  between train_fit and train_eval is possible; it never leaks into
  ``val_eval``, and ``val_eval`` is the reported number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from la_proteina.proteinfoundation.utils.cluster_utils import split_dataframe
from quality_graft.training import metrics


ProbeVariant = str

PLDDT_VARIANTS: tuple[str, ...] = ("s_only", "s_latents", "z_pooled", "s_z")


def pool_pair_features(z: Tensor, mask: Tensor) -> Tensor:
    """Masked pooling of the pair representation over the second axis.

    Args:
        z: ``[b, n, n, d_pair]`` pair representation.
        mask: ``[b, n]`` boolean (or float) residue mask. Padded ``j``
            columns are excluded from the mean and max reductions.

    Returns:
        ``[b, n, 3 * d_pair]`` = concat over the feature axis of
        ``(mean_j, max_j, diag)``. ``mean_j`` / ``max_j`` reduce over
        valid ``j`` only; a residue ``i`` with no valid ``j`` yields a
        zeroed row (no NaN, no ``-inf``). ``diag`` is ``z[:, k, k, :]``.
    """
    z = z.float()
    b, n, _, d_pair = z.shape
    pair_valid = (mask[:, :, None].bool() & mask[:, None, :].bool()).to(z.dtype)

    denom = pair_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean_j = (z * pair_valid[..., None]).sum(dim=2) / denom

    neg_inf = torch.finfo(z.dtype).min
    z_masked = z.masked_fill(~pair_valid[..., None].bool(), neg_inf)
    max_j = z_masked.max(dim=2).values
    row_has_valid = pair_valid.any(dim=2)
    max_j = torch.where(row_has_valid[..., None], max_j, torch.zeros_like(max_j))

    idx = torch.arange(n, device=z.device)
    diag = z[:, idx, idx, :]

    return torch.cat([mean_j, max_j, diag], dim=-1).float()


def split_swissprot_for_probe(
    df_data: pd.DataFrame,
    *,
    train_val_test: tuple[float, float, float] = (0.94, 0.03, 0.03),
    split_seed: int = 42,
    probe_seed: int = 0,
    train_split: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split SwissProt metadata into (train_fit, train_eval, val_eval).

    The train/val/test boundary reuses the QG datasplitter's
    ``split_dataframe(df, ["train", "val", "test"], train_val_test,
    seed=split_seed)`` verbatim so the probe's ``val_eval`` partition is
    exactly the training run's held-out val split (default ``seed=42``).
    The train side is then partitioned into a fit and an eval frame by a
    seeded random shuffle (``probe_seed``); ``train_split`` fraction goes
    to fit.

    Unlike the Complexa split, this carve is **not** cluster-disjoint —
    SwissProt has no cluster column, so family leak between ``train_fit``
    and ``train_eval`` is possible. It never leaks into ``val_eval``,
    which is the reported number.

    Args:
        df_data: SwissProt metadata frame (one row per structure).
        train_val_test: (train, val, test) fractions summing to 1.
        split_seed: seed forwarded to ``split_dataframe`` for the QG
            train/val/test boundary.
        probe_seed: seed for the within-train fit/eval random carve.
        train_split: fraction of the train frame assigned to fit.

    Returns:
        ``(train_fit_df, train_eval_df, val_eval_df)``. The union of the
        first two equals the ``split_dataframe`` train frame; the third
        equals its val frame.
    """
    splits = split_dataframe(
        df_data, ["train", "val", "test"], list(train_val_test), seed=split_seed
    )
    train_df = splits["train"].reset_index(drop=True)
    val_eval_df = splits["val"].reset_index(drop=True)

    rng = np.random.default_rng(probe_seed)
    perm = rng.permutation(len(train_df))
    n_fit = int(len(train_df) * train_split)
    fit_idx = perm[:n_fit]
    eval_idx = perm[n_fit:]
    train_fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
    train_eval_df = train_df.iloc[eval_idx].reset_index(drop=True)
    return train_fit_df, train_eval_df, val_eval_df


def build_probe_features(
    inter: dict[str, Tensor],
    *,
    variant: ProbeVariant,
) -> tuple[Tensor, Tensor, Tensor]:
    """Assemble ``(X, y_bin, mask_eff)`` for a pLDDT probe variant.

    ``inter`` carries the frozen-wrapper output plus the aligned bin
    labels and label mask: ``trunk_seqs`` ``[b, n, 768]``,
    ``trunk_pair`` ``[b, n, n, 256]``, ``local_latents`` ``[b, n, 8]``,
    ``mask`` ``[b, n]``, ``plddt_bin`` ``[b, n]``, ``plddt_mask``
    ``[b, n]``.

    Args:
        inter: wrapper-output dict extended with aligned pLDDT labels.
        variant: one of ``{"s_only", "s_latents", "z_pooled", "s_z"}``.

    Returns:
        ``X`` ``[b, n, d]`` (fp32), ``y_bin`` ``[b, n]``, ``mask_eff``
        ``[b, n]``. Raises ``ValueError`` if the features and labels
        disagree on the residue axis or if ``variant`` is unknown.
    """
    mask = inter["mask"]
    if mask.dtype != torch.bool:
        mask = mask.bool()

    s = inter["trunk_seqs"].float()
    z = inter["trunk_pair"].float()

    if variant == "s_only":
        X = s
    elif variant == "s_latents":
        latents = inter["local_latents"].float()
        X = torch.cat([s, latents], dim=-1)
    elif variant == "z_pooled":
        X = pool_pair_features(z, mask)
    elif variant == "s_z":
        X = torch.cat([s, pool_pair_features(z, mask)], dim=-1)
    else:
        raise ValueError(f"unknown probe variant {variant!r}")

    y_bin = inter["plddt_bin"]
    plddt_mask = inter["plddt_mask"]
    if plddt_mask.dtype != torch.bool:
        plddt_mask = plddt_mask.bool()

    if not (X.shape[1] == y_bin.shape[1] == mask.shape[1]):
        raise ValueError(
            f"{variant}: feature axis-1 {X.shape[1]} / label axis-1 "
            f"{y_bin.shape[1]} / mask axis-1 {mask.shape[1]} disagree"
        )

    mask_eff = (mask & plddt_mask).to(torch.float32)
    return X.float(), y_bin, mask_eff


class LinearProbe(nn.Module):
    """A single ``nn.Linear(d_in, n_out)`` mapping frozen features to logits."""

    def __init__(self, d_in: int, n_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, n_out)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features.float())


@dataclass
class ProbePartition:
    """One (features, bin labels, mask, bin count) partition for a probe.

    ``X``, ``y_bin``, and ``mask`` are aligned on the leading ``[b, n]``
    axes. ``num_bins`` is the pLDDT bin count used to form continuous
    predictions/targets for the correlation metrics.
    """

    X: Tensor
    y_bin: Tensor
    mask: Tensor
    num_bins: int


@dataclass
class ProbeData:
    """The three partitions the probe fit and evaluation consume.

    ``train_fit`` is optimised; ``train_eval`` and ``val_eval`` are
    held-out evaluation partitions (within-train and val respectively).
    All three share the same feature dimensionality and bin count.
    """

    train_fit: ProbePartition
    train_eval: ProbePartition
    val_eval: ProbePartition


def probe_metrics(
    logits: Tensor,
    y_bin: Tensor,
    mask: Tensor,
    num_bins: int,
) -> dict[str, float]:
    """Per-protein Pearson / Spearman of the probe's continuous EV.

    Forms continuous predictions via
    ``metrics._logits_to_continuous(logits, num_bins)`` and continuous
    targets via ``metrics._labels_to_continuous(y_bin, num_bins)``, then
    calls ``metrics.pearson_r`` / ``metrics.spearman_r`` (per-protein
    averaged) on the masked tensors. Does not reimplement the
    correlation math.

    Returns:
        ``{"pearson": float, "spearman": float}``.
    """
    pred_cont = metrics._logits_to_continuous(logits, num_bins)
    target_cont = metrics._labels_to_continuous(y_bin, num_bins)
    pearson = metrics.pearson_r(pred_cont, target_cont, mask)
    spearman = metrics.spearman_r(pred_cont, target_cont, mask)
    return {"pearson": float(pearson), "spearman": float(spearman)}


def fit_probe(
    features: ProbeData,
    *,
    steps: int,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    seed: int = 0,
    log_every: int = 100,
) -> list[dict]:
    """Gradient-descent fit of a ``LinearProbe`` on frozen features.

    Only the ``LinearProbe`` parameters are optimised (CE over the bin
    targets of ``features.train_fit``); the feature tensors are leaf
    constants with ``requires_grad=False``. Deterministic under ``seed``.

    Args:
        features: the three partitions (fit / train-eval / val-eval).
        steps: number of optimisation steps.
        lr: peak learning rate.
        warmup_steps: linear LR warmup length.
        seed: RNG seed for probe init and any stochastic op.
        log_every: emit a record every ``log_every`` steps (always at
            step 0 and the final step).

    Returns:
        A list of per-logged-step records, each with keys
        ``{"step", "train_fit_pearson", "train_eval_pearson",
        "val_eval_pearson"}`` (per-protein-averaged Pearson on each
        partition at that step).
    """
    torch.manual_seed(seed)

    fit = features.train_fit
    d_in = fit.X.shape[-1]
    n_out = int(fit.num_bins)
    probe = LinearProbe(d_in=d_in, n_out=n_out)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    fit_X = fit.X.detach().float()
    fit_y = fit.y_bin.long()
    fit_mask = fit.mask.bool()

    def eval_records(step: int) -> dict:
        probe.eval()
        with torch.no_grad():
            record: dict = {"step": step}
            for name, part in (
                ("train_fit", features.train_fit),
                ("train_eval", features.train_eval),
                ("val_eval", features.val_eval),
            ):
                logits = probe(part.X.float())
                m = probe_metrics(
                    logits, part.y_bin, part.mask.bool(), part.num_bins
                )
                record[f"{name}_pearson"] = m["pearson"]
        probe.train()
        return record

    valid = fit_mask.reshape(-1)
    flat_y = fit_y.reshape(-1)[valid]

    trace: list[dict] = [eval_records(0)]

    for step in range(1, steps + 1):
        probe.train()
        optimizer.zero_grad()
        logits = probe(fit_X)
        flat_logits = logits.reshape(-1, n_out)[valid]
        loss = F.cross_entropy(flat_logits, flat_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == steps:
            trace.append(eval_records(step))

    return trace


def ridge_probe_ev(
    X: Tensor,
    y_cont: Tensor,
    mask: Tensor,
    *,
    alpha: float = 1.0,
) -> Tensor:
    """Closed-form ridge regression on the continuous expected value.

    A cross-check for the GD probe: fits ``w`` minimising
    ``sum_masked (X w - y_cont)^2 + alpha * ||w||^2`` and returns the
    masked predictions ``X w`` (``[b, n]``, matching ``mask``).

    Args:
        X: ``[..., d]`` fp32 features flattened over the leading axes.
        y_cont: continuous target aligned with ``mask``.
        mask: validity mask over the leading axes.
        alpha: ridge penalty.

    Returns:
        Masked continuous predictions aligned with ``mask``.
    """
    d = X.shape[-1]
    X_flat = X.reshape(-1, d).to(torch.float64)
    y_flat = y_cont.reshape(-1).to(torch.float64)
    valid = mask.reshape(-1).bool()

    X_v = X_flat[valid]
    y_v = y_flat[valid]

    ones = torch.ones(X_v.shape[0], 1, dtype=torch.float64, device=X_v.device)
    X_aug = torch.cat([X_v, ones], dim=-1)
    d_aug = X_aug.shape[-1]

    gram = X_aug.T @ X_aug
    reg = alpha * torch.eye(d_aug, dtype=torch.float64, device=X_aug.device)
    reg[-1, -1] = 0.0
    rhs = X_aug.T @ y_v
    w = torch.linalg.solve(gram + reg, rhs)

    ones_all = torch.ones(
        X_flat.shape[0], 1, dtype=torch.float64, device=X_flat.device
    )
    preds_flat = (torch.cat([X_flat, ones_all], dim=-1) @ w).to(torch.float32)
    preds = preds_flat.reshape(mask.shape)
    return preds * mask.to(preds.dtype)
