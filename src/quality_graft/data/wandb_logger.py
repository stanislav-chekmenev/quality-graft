"""W&B logging utilities for the dataset generation pipeline.

All public functions are no-ops when W&B is not initialized
(guard on ``wandb.run is not None``).
"""

import argparse
import logging
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment analysis helpers
# ---------------------------------------------------------------------------

def longest_contiguous_below(plddt: np.ndarray, threshold: float) -> int:
    """Length of the longest contiguous run of residues with pLDDT < threshold."""
    below = plddt < threshold
    if not below.any():
        return 0
    changes = np.diff(below.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int((ends - starts).max())


def count_segments_below(plddt: np.ndarray, threshold: float) -> int:
    """Number of contiguous segments with pLDDT < threshold."""
    below = plddt < threshold
    if not below.any():
        return 0
    changes = np.diff(below.astype(int), prepend=0, append=0)
    return int((changes == 1).sum())


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_protein_metrics(
    pdb_id: str,
    plddt: np.ndarray,
    n_residues: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Compute all per-protein metrics from a pLDDT array.

    Returns a dict with keys matching the ``wandb.log`` format:
        protein/pdb_id, protein/length, protein/mean_plddt, protein/median_plddt,
        protein/p10_plddt, protein/p25_plddt, protein/std_plddt, protein/iqr_plddt,
        protein/frac_ge90, protein/frac_ge70, protein/frac_lt50,
        protein/L70, protein/L70_frac,
        protein/longest_low_segment, protein/num_low_segments,
        protein/nterm_30_mean, protein/cterm_30_mean, protein/core_mean,
        protein/boltz_walltime_s.
    """
    metrics: dict[str, Any] = {
        "protein/pdb_id": pdb_id,
        "protein/length": n_residues,
        "protein/mean_plddt": float(plddt.mean()),
        "protein/median_plddt": float(np.median(plddt)),
        "protein/p10_plddt": float(np.percentile(plddt, 10)),
        "protein/p25_plddt": float(np.percentile(plddt, 25)),
        "protein/std_plddt": float(plddt.std()),
        "protein/iqr_plddt": float(
            np.percentile(plddt, 75) - np.percentile(plddt, 25)
        ),
        "protein/frac_ge90": float((plddt >= 0.90).mean()),
        "protein/frac_ge70": float((plddt >= 0.70).mean()),
        "protein/frac_lt50": float((plddt < 0.50).mean()),
        "protein/L70": int((plddt >= 0.70).sum()),
        "protein/L70_frac": float((plddt >= 0.70).mean()),
        "protein/longest_low_segment": longest_contiguous_below(plddt, 0.50),
        "protein/num_low_segments": count_segments_below(plddt, 0.50),
        "protein/nterm_30_mean": (
            float(plddt[:30].mean()) if len(plddt) >= 30 else float(plddt.mean())
        ),
        "protein/cterm_30_mean": (
            float(plddt[-30:].mean()) if len(plddt) >= 30 else float(plddt.mean())
        ),
        "protein/core_mean": (
            float(plddt[30:-30].mean()) if len(plddt) > 60 else float(plddt.mean())
        ),
        "protein/boltz_walltime_s": elapsed_s,
    }
    return metrics


# ---------------------------------------------------------------------------
# W&B lifecycle
# ---------------------------------------------------------------------------

def init_wandb_run(args: argparse.Namespace) -> None:
    """Initialize a W&B run with config derived from CLI args.

    No-op if ``args.no_wandb`` is ``True`` (or the attribute is missing).
    """
    if getattr(args, "no_wandb", True):
        return

    import wandb

    wandb.init(
        project=getattr(args, "wandb_project", "quality-graft"),
        name=getattr(args, "wandb_run_name", None),
        entity=getattr(args, "wandb_entity", None),
        job_type="dataset-generation",
        config={
            "model": getattr(args, "model", "boltz1"),
            "diffusion_samples": getattr(args, "diffusion_samples", 1),
            "sampling_steps": getattr(args, "sampling_steps", 200),
            "recycling_steps": getattr(args, "recycling_steps", 3),
            "use_msa_server": getattr(args, "use_msa_server", False),
            "num_plddt_bins": getattr(args, "num_bins", 50),
            "accelerator": getattr(args, "accelerator", "gpu"),
            "input_dir": str(getattr(args, "input_dir", "")),
        },
    )


# ---------------------------------------------------------------------------
# Per-protein logging
# ---------------------------------------------------------------------------

def log_protein_metrics(
    pdb_id: str,
    plddt: np.ndarray,
    n_residues: int,
    elapsed_s: float,
    n_processed: int,
    n_failed: int,
    n_skipped: int,
) -> dict[str, Any]:
    """Compute and log per-protein metrics to W&B.

    Returns the metrics dict (useful for accumulating stats).
    Logs to W&B only if ``wandb.run is not None``.
    """
    metrics = compute_protein_metrics(pdb_id, plddt, n_residues, elapsed_s)
    metrics["progress/n_processed"] = n_processed
    metrics["progress/n_failed"] = n_failed
    metrics["progress/n_skipped"] = n_skipped

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass

    # Store raw plddt array for summary plots (not logged to W&B per-step)
    metrics["_plddt_array"] = plddt

    return metrics


# ---------------------------------------------------------------------------
# Dataset summary (plots + table)
# ---------------------------------------------------------------------------

def log_dataset_summary(protein_stats: list[dict]) -> None:
    """Generate and log all summary plots and a W&B Table.

    No-op if ``wandb.run is None``.
    ``protein_stats`` is a list of dicts (one per protein), each produced by
    :func:`compute_protein_metrics`.
    """
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return

    if not protein_stats:
        return

    # ------------------------------------------------------------------
    # Extract arrays from stats
    # ------------------------------------------------------------------
    means = np.array([s["protein/mean_plddt"] for s in protein_stats])
    medians = np.array([s["protein/median_plddt"] for s in protein_stats])
    lengths = np.array([s["protein/length"] for s in protein_stats])
    p10s = np.array([s["protein/p10_plddt"] for s in protein_stats])
    p25s = np.array([s["protein/p25_plddt"] for s in protein_stats])
    f90s = np.array([s["protein/frac_ge90"] for s in protein_stats])
    f70s = np.array([s["protein/frac_ge70"] for s in protein_stats])
    f50s = np.array([s["protein/frac_lt50"] for s in protein_stats])
    l70_fracs = np.array([s["protein/L70_frac"] for s in protein_stats])
    stds = np.array([s["protein/std_plddt"] for s in protein_stats])
    iqrs = np.array([s["protein/iqr_plddt"] for s in protein_stats])
    nterm_means = np.array([s["protein/nterm_30_mean"] for s in protein_stats])
    cterm_means = np.array([s["protein/cterm_30_mean"] for s in protein_stats])
    core_means = np.array([s["protein/core_mean"] for s in protein_stats])
    longest_segs = np.array([s["protein/longest_low_segment"] for s in protein_stats])
    num_segs = np.array([s["protein/num_low_segments"] for s in protein_stats])

    # Extract raw plddt arrays (stored by log_protein_metrics)
    raw_plddts = [s["_plddt_array"] for s in protein_stats if "_plddt_array" in s]

    images: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Category 1: pLDDT Distributions
    # ------------------------------------------------------------------

    # Plot 1: Per-protein mean pLDDT histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(means, bins=50, edgecolor="black", alpha=0.7)
    for cutoff, color, label in [
        (0.5, "red", "0.50"),
        (0.7, "orange", "0.70"),
        (0.9, "green", "0.90"),
    ]:
        ax.axvline(cutoff, color=color, linestyle="--", label=label)
    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("Count")
    ax.set_title("Per-Protein Mean pLDDT Distribution")
    ax.legend()
    plt.tight_layout()
    images["distributions/mean_plddt_hist"] = wandb.Image(fig)
    plt.close(fig)

    # Plot 2: Per-protein median pLDDT histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(medians, bins=50, edgecolor="black", alpha=0.7)
    for cutoff, color, label in [
        (0.5, "red", "0.50"),
        (0.7, "orange", "0.70"),
        (0.9, "green", "0.90"),
    ]:
        ax.axvline(cutoff, color=color, linestyle="--", label=label)
    ax.set_xlabel("Median pLDDT")
    ax.set_ylabel("Count")
    ax.set_title("Per-Protein Median pLDDT Distribution")
    ax.legend()
    plt.tight_layout()
    images["distributions/median_plddt_hist"] = wandb.Image(fig)
    plt.close(fig)

    # Plot 3: ECDF of per-protein mean pLDDT
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_means = np.sort(means)
    ecdf = np.arange(1, len(sorted_means) + 1) / len(sorted_means)
    ax.plot(sorted_means, ecdf, linewidth=2)
    for cutoff, color in [(0.5, "red"), (0.7, "orange"), (0.9, "green")]:
        ax.axvline(cutoff, color=color, linestyle="--", alpha=0.7)
    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("Fraction of proteins")
    ax.set_title("ECDF of Per-Protein Mean pLDDT")
    plt.tight_layout()
    images["distributions/mean_plddt_ecdf"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: Per-protein 10th/25th percentile histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(p10s, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("10th Percentile pLDDT")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Protein 10th Percentile pLDDT")
    axes[1].hist(p25s, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("25th Percentile pLDDT")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-Protein 25th Percentile pLDDT")
    plt.tight_layout()
    images["distributions/percentile_hist"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: Pooled per-residue pLDDT histogram
    if raw_plddts:
        pooled = np.concatenate(raw_plddts)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pooled, bins=100, edgecolor="black", alpha=0.7)
        for cutoff, color, label in [
            (0.5, "red", "0.50"),
            (0.7, "orange", "0.70"),
            (0.9, "green", "0.90"),
        ]:
            ax.axvline(cutoff, color=color, linestyle="--", label=label)
        ax.set_xlabel("pLDDT")
        ax.set_ylabel("Count")
        ax.set_title("Pooled Per-Residue pLDDT Distribution")
        ax.legend()
        plt.tight_layout()
        images["distributions/pooled_residue_plddt_hist"] = wandb.Image(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Category 2: Confidence Coverage
    # ------------------------------------------------------------------

    # Plot 4: f90 / f70 / f50 distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(
        axes,
        [f90s, f70s, f50s],
        ["f90 (>=0.90)", "f70 (>=0.70)", "f50 (<0.50)"],
    ):
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Count")
        ax.set_title(title)
    plt.tight_layout()
    images["coverage/fraction_distributions"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: L70/L distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(l70_fracs, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("L70 / L (fraction of residues >= 0.70)")
    ax.set_ylabel("Count")
    ax.set_title("L70/L Distribution Across Proteins")
    plt.tight_layout()
    images["coverage/l70_frac_hist"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: Stacked bar chart (per-protein confidence breakdown)
    if len(means) <= 200:
        sort_idx = np.argsort(means)
    else:
        # Subsample for readability
        sort_idx = np.argsort(means)[:: max(1, len(means) // 200)]
    fig, ax = plt.subplots(figsize=(max(8, len(sort_idx) * 0.05), 5))
    x = np.arange(len(sort_idx))
    s_f90 = f90s[sort_idx]
    s_f70_90 = f70s[sort_idx] - f90s[sort_idx]
    s_f50_70 = (1 - f50s[sort_idx]) - f70s[sort_idx]
    s_lt50 = f50s[sort_idx]
    ax.bar(x, s_f90, label=">=0.90", color="green")
    ax.bar(x, s_f70_90, bottom=s_f90, label="0.70-0.90", color="yellowgreen")
    ax.bar(x, s_f50_70, bottom=s_f90 + s_f70_90, label="0.50-0.70", color="orange")
    ax.bar(x, s_lt50, bottom=s_f90 + s_f70_90 + s_f50_70, label="<0.50", color="red")
    ax.set_xlabel("Proteins (sorted by mean pLDDT)")
    ax.set_ylabel("Fraction")
    ax.set_title("Per-Protein Confidence Breakdown")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, len(sort_idx) - 0.5)
    plt.tight_layout()
    images["coverage/stacked_bar"] = wandb.Image(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Category 3: Relationship Plots
    # ------------------------------------------------------------------

    # Plot 5: Length vs mean pLDDT
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, means, alpha=0.5, s=10)
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("Mean pLDDT")
    ax.set_title("Protein Length vs Mean pLDDT")
    plt.tight_layout()
    images["relationships/length_vs_mean"] = wandb.Image(fig)
    plt.close(fig)

    # Plot 6: Mean vs std pLDDT (coloured by length)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        means, stds, c=np.log10(lengths + 1), alpha=0.5, s=10, cmap="viridis"
    )
    plt.colorbar(scatter, ax=ax, label="log10(length)")
    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("Std pLDDT")
    ax.set_title("Mean vs Std pLDDT (colored by length)")
    plt.tight_layout()
    images["relationships/mean_vs_std"] = wandb.Image(fig)
    plt.close(fig)

    # Plot 7: Mean vs IQR pLDDT
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(means, iqrs, alpha=0.5, s=10)
    ax.set_xlabel("Mean pLDDT")
    ax.set_ylabel("IQR pLDDT")
    ax.set_title("Mean vs IQR pLDDT")
    plt.tight_layout()
    images["relationships/mean_vs_iqr"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: Length vs f70
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, f70s, alpha=0.5, s=10)
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("f70 (fraction >= 0.70)")
    ax.set_title("Protein Length vs f70")
    plt.tight_layout()
    images["relationships/length_vs_f70"] = wandb.Image(fig)
    plt.close(fig)

    # Plot: Length vs f50
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, f50s, alpha=0.5, s=10)
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("f50 (fraction < 0.50)")
    ax.set_title("Protein Length vs f50")
    plt.tight_layout()
    images["relationships/length_vs_f50"] = wandb.Image(fig)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Category 4: Positional Summaries
    # ------------------------------------------------------------------

    if raw_plddts:
        # Plot: Average pLDDT vs relative position
        n_pos_bins = 100
        pos_sums = np.zeros(n_pos_bins)
        pos_counts = np.zeros(n_pos_bins)
        for plddt_arr in raw_plddts:
            L = len(plddt_arr)
            if L == 0:
                continue
            rel_pos = np.arange(L) / L
            bin_idx = np.clip((rel_pos * n_pos_bins).astype(int), 0, n_pos_bins - 1)
            for b in range(n_pos_bins):
                mask = bin_idx == b
                if mask.any():
                    pos_sums[b] += plddt_arr[mask].sum()
                    pos_counts[b] += mask.sum()
        pos_means = np.divide(
            pos_sums, pos_counts, out=np.zeros_like(pos_sums), where=pos_counts > 0
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.linspace(0, 1, n_pos_bins), pos_means, linewidth=2)
        ax.set_xlabel("Relative Position (N-term → C-term)")
        ax.set_ylabel("Mean pLDDT")
        ax.set_title("Average pLDDT vs Relative Position")
        plt.tight_layout()
        images["positional/avg_plddt_vs_position"] = wandb.Image(fig)
        plt.close(fig)

        # Plot: N-term / Core / C-term box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(
            [nterm_means, core_means, cterm_means],
            labels=["N-term (first 30)", "Core", "C-term (last 30)"],
        )
        ax.set_ylabel("Mean pLDDT")
        ax.set_title("N-term / Core / C-term pLDDT Comparison")
        plt.tight_layout()
        images["positional/nterm_core_cterm_boxplot"] = wandb.Image(fig)
        plt.close(fig)

        # Plot: pLDDT heatmap (proteins × relative position)
        n_heatmap_bins = 100
        heatmap_rows = []
        for plddt_arr in raw_plddts:
            L = len(plddt_arr)
            if L == 0:
                continue
            interp_x = np.linspace(0, L - 1, n_heatmap_bins)
            row = np.interp(interp_x, np.arange(L), plddt_arr)
            heatmap_rows.append(row)
        if heatmap_rows:
            heatmap = np.array(heatmap_rows)
            # Sort by mean pLDDT
            sort_idx_hm = np.argsort(heatmap.mean(axis=1))
            heatmap = heatmap[sort_idx_hm]
            fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_rows) * 0.04)))
            im = ax.imshow(
                heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                extent=[0, 1, 0, len(heatmap_rows)],
            )
            ax.set_xlabel("Relative Position")
            ax.set_ylabel("Proteins (sorted by mean pLDDT)")
            ax.set_title("pLDDT Heatmap")
            plt.colorbar(im, ax=ax, label="pLDDT")
            plt.tight_layout()
            images["positional/plddt_heatmap"] = wandb.Image(fig)
            plt.close(fig)

    # ------------------------------------------------------------------
    # Category 5: Segment Statistics
    # ------------------------------------------------------------------

    # Plot 8: Longest low-confidence segment distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(longest_segs, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Longest Low-Confidence Segment (pLDDT < 0.50)")
    ax.set_ylabel("Count")
    ax.set_title("Longest Low-Confidence Segment Distribution")
    plt.tight_layout()
    images["segments/longest_low_segment_hist"] = wandb.Image(fig)
    plt.close(fig)

    # Plot 9: Number of low-confidence segments
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(num_segs, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Low-Confidence Segments")
    ax.set_ylabel("Count")
    ax.set_title("Number of Low-Confidence Segments Distribution")
    plt.tight_layout()
    images["segments/num_low_segments_hist"] = wandb.Image(fig)
    plt.close(fig)

    wandb.log(images)

    # ------------------------------------------------------------------
    # W&B Summary Table
    # ------------------------------------------------------------------
    columns = [
        "pdb_id", "length", "mean_plddt", "median_plddt",
        "p10_plddt", "p25_plddt", "std_plddt", "iqr_plddt",
        "f90", "f70", "f50", "L70", "L70_frac",
        "longest_low_seg", "num_low_segs",
        "nterm_30_mean", "cterm_30_mean", "core_mean",
        "walltime_s",
    ]
    rows = []
    for s in protein_stats:
        rows.append([
            s["protein/pdb_id"], s["protein/length"],
            s["protein/mean_plddt"], s["protein/median_plddt"],
            s["protein/p10_plddt"], s["protein/p25_plddt"],
            s["protein/std_plddt"], s["protein/iqr_plddt"],
            s["protein/frac_ge90"], s["protein/frac_ge70"],
            s["protein/frac_lt50"], s["protein/L70"], s["protein/L70_frac"],
            s["protein/longest_low_segment"], s["protein/num_low_segments"],
            s["protein/nterm_30_mean"], s["protein/cterm_30_mean"],
            s["protein/core_mean"], s["protein/boltz_walltime_s"],
        ])
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"dataset/protein_table": table})


def finish_wandb_run() -> None:
    """Finalize W&B run. No-op if ``wandb.run is None``."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
