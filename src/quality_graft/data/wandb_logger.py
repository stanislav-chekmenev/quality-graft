"""W&B logging utilities for dataset preprocessing.

Logs a single dataset-level summary at the end of preprocessing:
histogram of per-protein mean pLDDT + scalar stats (mean, std, max, min, count).

All public functions are no-ops when W&B is not initialized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def collect_dataset_stats(processed_dir: Path) -> list[dict[str, Any]]:
    """Scan .pt files and collect per-protein mean pLDDT + residue count.

    Returns a list of dicts with keys: structure_id, mean_plddt, n_residues.
    Skips files without pLDDT labels.
    """
    import torch

    processed_dir = Path(processed_dir)
    pt_files = sorted(processed_dir.glob("*.pt"))
    stats: list[dict[str, Any]] = []

    for pt_path in pt_files:
        graph = torch.load(pt_path, weights_only=False)
        if not hasattr(graph, "plddt") or graph.plddt is None:
            continue

        plddt_np = graph.plddt.numpy()
        stats.append({
            "structure_id": pt_path.stem,
            "mean_plddt": float(plddt_np.mean()),
            "n_residues": int(plddt_np.shape[0]),
        })

    return stats


def log_dataset_summary(protein_stats: list[dict[str, Any]]) -> None:
    """Log dataset-level pLDDT summary to W&B.

    Logs: histogram of per-protein mean pLDDT, mean, std, max, min, count.
    No-op if wandb.run is None or stats is empty.
    """
    if wandb is None or wandb.run is None:
        return

    if not protein_stats:
        return

    means = np.array([s["mean_plddt"] for s in protein_stats])

    wandb.log({
        "dataset/plddt_histogram": wandb.Histogram(means),
        "dataset/mean_plddt": float(means.mean()),
        "dataset/std_plddt": float(means.std()),
        "dataset/max_plddt": float(means.max()),
        "dataset/min_plddt": float(means.min()),
        "dataset/num_proteins": len(protein_stats),
    })
