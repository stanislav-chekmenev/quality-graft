"""Frozen-feature linear-probe diagnostic harness for the La-Proteina trunk.

Answers one question: are the frozen La-Proteina features (``trunk_seqs``,
``local_latents``, ``trunk_pair``) so near-linearly predictive of the AF2
B-factor pLDDT bin labels that a single ``nn.Linear`` reaches high Pearson
on the *held-out* val split? A high ``val_eval`` Pearson from a linear
probe alone would explain a distillation run's fast convergence as
feature-linearity, not a val-split leak.

Loads the frozen ``LaProteinaWrapper`` from the model config's checkpoint
paths with the pinned ``t_value = 0.99`` and ``deterministic_encode =
True``; features are cast to fp32 (bf16 would add Pearson noise). The
train/val boundary reuses the QG random ``split_dataframe`` split
(seed 42, ``split_swissprot_for_probe``); the train side is further
partitioned into a seeded random fit/eval carve (NOT cluster-disjoint —
SwissProt has no cluster column).

Run recipe (GPU recommended; CPU works but slow):

    python scripts/run_frozen_feature_probe.py \
      data=swissprot data.local_only=true \
      model/la_proteina_wrapper=la_proteina_wrapper \
      +probe.max_proteins_train_fit=300 +probe.max_proteins_train_eval=200 \
      +probe.max_proteins_val_eval=200 +probe.fit_steps=3000 \
      +probe.log_every=100 +probe.seed=0 +probe.device=auto \
      +probe.t_value=0.99 +probe.deterministic_encode=true
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Mirror scripts/train.py's sys.path setup so la_proteina / quality_graft
# import cleanly before any project imports below.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LA_PROTEINA_DIR = SRC_DIR / "la_proteina"
for _p in [PROJECT_ROOT, SRC_DIR, LA_PROTEINA_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.la_proteina.proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    DensePaddingDataLoader,
)
from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper  # noqa: E402
from quality_graft.probes.frozen_feature_probe import (  # noqa: E402
    PLDDT_VARIANTS,
    ProbeData,
    ProbePartition,
    build_probe_features,
    fit_probe,
    ridge_probe_ev,
    split_swissprot_for_probe,
)
from quality_graft.training.metrics import _labels_to_continuous, pearson_r  # noqa: E402

# The train.py script builds the datamodule directly (not via Hydra
# instantiate), so we reuse its builder to stay in lockstep with the
# training run's data pipeline.
import scripts.train as train_script  # noqa: E402

from quality_graft.data.plddt_utils import NUM_PLDDT_BINS  # noqa: E402


def _gate_loguru_to_rank0() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    if local_rank != 0 or node_rank != 0 or global_rank != 0:
        logger.remove()


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = OmegaConf.select(cfg, "probe.device", default="auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _build_wrapper(cfg: DictConfig, device: torch.device) -> LaProteinaWrapper:
    lp_cfg = cfg.model.la_proteina_wrapper
    t_value = float(OmegaConf.select(cfg, "probe.t_value", default=0.99))
    deterministic = bool(
        OmegaConf.select(cfg, "probe.deterministic_encode", default=True)
    )
    wrapper = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=lp_cfg.proteina_ckpt_path,
        autoencoder_ckpt_path=lp_cfg.autoencoder_ckpt_path,
        device=str(device),
        use_decoder=False,
        t_value=t_value,
        deterministic_encode=deterministic,
    )
    assert abs(wrapper.t_value - 0.99) < 1e-9, (
        f"expected pinned t_value=0.99, got {wrapper.t_value}"
    )
    return wrapper.to(device).eval()


def _partition_pt_name(row) -> str:
    """Reconstruct the .pt filename for a split row.

    Mirrors ``PDBLightningDataModule._get_dataset`` exactly: a ``chain``
    column present yields ``{pdb}_{chain}.pt`` (SwissProt local_only
    splits the stem on the first ``_``, so ``pdb`` is missing the ``_v4``
    suffix and the chain restores it), otherwise ``{pdb}.pt``.
    """
    pdb = row["pdb"]
    chain = row.get("chain")
    if chain is not None and not (isinstance(chain, float) and pd.isna(chain)):
        return f"{pdb}_{chain}.pt"
    return f"{pdb}.pt"


def _filter_readable(
    partition_df,
    processed_dir: Path,
    cap: int,
) -> tuple["pd.DataFrame", int, int]:
    """Keep the first ``cap`` rows whose .pt file is a readable torch save.

    Roughly 10% of the staged SwissProt .pt files are empty legacy tars
    (valid tar header, no torch payload). ``zipfile.is_zipfile`` is an
    O(1) proxy for loadability on exactly this corruption mode: good
    saves are zip-format, the truncated ones are tar-only (verified 0
    mismatches vs ``torch.load`` over 150 files), so it avoids a second
    full ``torch.load`` here. Rows are scanned in the frame's existing
    (seed-42-derived) order; skipping never reshuffles.

    Returns ``(filtered_df, n_checked, n_skipped)`` where ``filtered_df``
    holds up to ``cap`` readable rows (fewer if the partition is
    exhausted first).
    """
    keep_positions: list[int] = []
    n_checked = 0
    n_skipped = 0
    for pos in range(len(partition_df)):
        if len(keep_positions) >= cap:
            break
        row = partition_df.iloc[pos]
        n_checked += 1
        path = processed_dir / _partition_pt_name(row)
        if zipfile.is_zipfile(path):
            keep_positions.append(pos)
        else:
            n_skipped += 1
    filtered = partition_df.iloc[keep_positions].reset_index(drop=True)
    return filtered, n_checked, n_skipped


def _partition_loader(datamodule, partition_df) -> DensePaddingDataLoader:
    datamodule.dfs_splits["train"] = partition_df
    dataset = datamodule._get_dataset("train")
    return DensePaddingDataLoader(
        dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _accumulate_features(
    wrapper: LaProteinaWrapper,
    loader: DensePaddingDataLoader,
    variants: tuple[str, ...],
    max_proteins: int,
    device: torch.device,
) -> dict[str, tuple[list, list, list]]:
    acc: dict[str, tuple[list, list, list]] = {v: ([], [], []) for v in variants}
    n_seen = 0
    for batch in loader:
        if n_seen >= max_proteins:
            break
        batch = batch.to(device)
        with torch.no_grad():
            reprs = wrapper(batch)  # populates batch["mask"] as a side effect
        mask = batch["mask"]
        inter = {
            "trunk_seqs": reprs["trunk_seqs"],
            "trunk_pair": reprs["trunk_pair"],
            "local_latents": reprs["local_latents"],
            "mask": mask,
            "plddt_bin": batch["plddt_bin"],
            "plddt_mask": mask,
        }
        for variant in variants:
            X, y_bin, mask_eff = build_probe_features(inter, variant=variant)
            acc[variant][0].append(X.float().cpu())
            acc[variant][1].append(y_bin.cpu())
            acc[variant][2].append(mask_eff.float().cpu())
        n_seen += int(mask.shape[0])
    return acc


def _pad_and_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_n = max(t.shape[1] for t in tensors)
    out: list[torch.Tensor] = []
    for t in tensors:
        n = t.shape[1]
        if n == max_n:
            out.append(t)
            continue
        pad = (0, 0, 0, max_n - n) if t.ndim == 3 else (0, max_n - n)
        out.append(torch.nn.functional.pad(t, pad))
    return torch.cat(out, dim=0)


def _build_partition(
    acc_variant: tuple[list, list, list], num_bins: int
) -> ProbePartition:
    X = _pad_and_stack(acc_variant[0])
    y = _pad_and_stack(acc_variant[1])
    mask = _pad_and_stack(acc_variant[2])
    return ProbePartition(X=X, y_bin=y.long(), mask=mask, num_bins=num_bins)


def _first_cross(trace: list[dict], key: str, threshold: float) -> str:
    for record in trace:
        if record[key] >= threshold:
            return str(record["step"])
    return "never"


def _report_variant(name: str, trace: list[dict]) -> None:
    final = trace[-1]
    cross = _first_cross(trace, "val_eval_pearson", 0.9)
    logger.info(
        f"[{name}] final Pearson  train_fit={final['train_fit_pearson']:.4f}  "
        f"train_eval={final['train_eval_pearson']:.4f}  "
        f"val_eval={final['val_eval_pearson']:.4f}  |  "
        f"val_eval first >= 0.9 at step: {cross}"
    )
    trace_str = "  ".join(
        f"{r['step']}:{r['val_eval_pearson']:.3f}" for r in trace
    )
    logger.info(f"[{name}] val_eval Pearson-vs-step: {trace_str}")


def _ridge_crosscheck(name: str, data: ProbeData) -> None:
    num_bins = data.train_fit.num_bins
    for pname, part in (
        ("train_fit", data.train_fit),
        ("train_eval", data.train_eval),
        ("val_eval", data.val_eval),
    ):
        target = _labels_to_continuous(part.y_bin, num_bins)
        preds = ridge_probe_ev(part.X, target, part.mask)
        r = pearson_r(preds, target, part.mask.bool())
        logger.info(f"[{name}][ridge] {pname} Pearson (self-fit) = {float(r):.4f}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _gate_loguru_to_rank0()

    seed = int(OmegaConf.select(cfg, "probe.seed", default=0))
    fit_steps = int(OmegaConf.select(cfg, "probe.fit_steps", default=3000))
    log_every = int(OmegaConf.select(cfg, "probe.log_every", default=100))
    cap_fit = int(OmegaConf.select(cfg, "probe.max_proteins_train_fit", default=300))
    cap_train_eval = int(
        OmegaConf.select(cfg, "probe.max_proteins_train_eval", default=200)
    )
    cap_val_eval = int(
        OmegaConf.select(cfg, "probe.max_proteins_val_eval", default=200)
    )

    device = _resolve_device(cfg)
    wrapper = _build_wrapper(cfg, device)
    logger.info(f"Probe device = {device}")
    logger.info(f"Pinned t_value = {wrapper.t_value}")
    logger.info(f"deterministic_encode = {wrapper.deterministic_encode}")

    datamodule = train_script._build_swissprot_data_module(cfg.data)
    datamodule.num_workers = 0
    datamodule.setup("fit")

    train_fit_df, train_eval_df, val_eval_df = split_swissprot_for_probe(
        datamodule.df_data,
        train_val_test=tuple(cfg.data.train_val_test),
        split_seed=42,
        probe_seed=seed,
    )
    logger.info(
        f"Split sizes  train_fit={len(train_fit_df)}  "
        f"train_eval={len(train_eval_df)}  val_eval={len(val_eval_df)}"
    )

    caps = {
        "train_fit": cap_fit,
        "train_eval": cap_train_eval,
        "val_eval": cap_val_eval,
    }
    raw_partition_dfs = {
        "train_fit": train_fit_df,
        "train_eval": train_eval_df,
        "val_eval": val_eval_df,
    }

    processed_dir = datamodule.processed_dir
    partition_dfs: dict[str, pd.DataFrame] = {}
    total_skipped = 0
    for pname, raw_df in raw_partition_dfs.items():
        filtered_df, n_checked, n_skipped = _filter_readable(
            raw_df, processed_dir, caps[pname]
        )
        partition_dfs[pname] = filtered_df
        total_skipped += n_skipped
        logger.info(
            f"{pname}: {len(filtered_df)}/{n_checked} readable (.pt), "
            f"skipped {n_skipped} unreadable"
        )

    logger.warning(
        "val_eval approximates the training-run val split; "
        f"{partition_dfs['val_eval'].shape[0]} readable val files were sampled "
        f"and unreadable files (staged-data truncation) were skipped."
    )

    acc_by_partition: dict[str, dict] = {}
    for pname, part_df in partition_dfs.items():
        loader = _partition_loader(datamodule, part_df)
        logger.info(f"Extracting frozen features: {pname} (cap {caps[pname]})")
        acc_by_partition[pname] = _accumulate_features(
            wrapper, loader, PLDDT_VARIANTS, caps[pname], device
        )

    for variant in PLDDT_VARIANTS:
        data = ProbeData(
            train_fit=_build_partition(
                acc_by_partition["train_fit"][variant], NUM_PLDDT_BINS
            ),
            train_eval=_build_partition(
                acc_by_partition["train_eval"][variant], NUM_PLDDT_BINS
            ),
            val_eval=_build_partition(
                acc_by_partition["val_eval"][variant], NUM_PLDDT_BINS
            ),
        )
        trace = fit_probe(
            data, steps=fit_steps, lr=1e-4, seed=seed, log_every=log_every
        )
        _report_variant(f"plddt/{variant}", trace)
        _ridge_crosscheck(f"plddt/{variant}", data)


if __name__ == "__main__":
    main()
