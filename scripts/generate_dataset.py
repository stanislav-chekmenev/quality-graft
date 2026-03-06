#!/usr/bin/env python
"""Generate pLDDT label dataset by running Boltz predictions on CIF files.

Usage:
    python scripts/generate_dataset.py --input-dir data/raw/structures/ --no-wandb
    python scripts/generate_dataset.py --single-cif data/raw/structures/1abc.cif --no-wandb
    python scripts/generate_dataset.py --input-dir data/raw/structures/ --wandb-project quality-graft
"""

from __future__ import annotations

import argparse
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import torch

from quality_graft.data.boltz_runner import run_boltz_predict
from quality_graft.data.cif_utils import ChainInfo, chains_to_boltz_yaml, parse_cif_chains
from quality_graft.data.plddt_utils import plddt_to_bin
from quality_graft.data.wandb_logger import (
    finish_wandb_run,
    init_wandb_run,
    log_dataset_summary,
    log_protein_metrics,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pLDDT label dataset from CIF files using Boltz predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output paths
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Path to directory with CIF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/labels/"),
        help="Path to save .pt label files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/processed/boltz_work/"),
        help="Working directory for Boltz intermediate files.",
    )

    # Boltz parameters
    parser.add_argument("--model", type=str, default="boltz1", help="Boltz model name.")
    parser.add_argument(
        "--use-msa-server", action="store_true", help="Enable MSA server for Boltz."
    )
    parser.add_argument(
        "--diffusion-samples", type=int, default=1, help="Number of diffusion samples."
    )
    parser.add_argument(
        "--sampling-steps", type=int, default=200, help="Number of sampling steps."
    )
    parser.add_argument(
        "--recycling-steps", type=int, default=3, help="Number of recycling steps."
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices.")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type."
    )

    # Processing options
    parser.add_argument(
        "--num-bins", type=int, default=50, help="Number of pLDDT bins."
    )
    parser.add_argument(
        "--override", action="store_true", help="Reprocess existing outputs."
    )
    parser.add_argument(
        "--single-cif",
        type=Path,
        default=None,
        help="Process a single CIF file path (for testing).",
    )

    # W&B options
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="quality-graft",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="W&B run name."
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/team."
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging."
    )

    args = parser.parse_args()

    # Validation
    if args.input_dir is None and args.single_cif is None:
        parser.error("Either --input-dir or --single-cif must be provided.")

    if args.single_cif is not None and not args.single_cif.is_file():
        parser.error(f"--single-cif file does not exist: {args.single_cif}")

    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist: {args.input_dir}")

    return args


def collect_cif_files(args: argparse.Namespace) -> list[Path]:
    """Collect CIF files from --input-dir or --single-cif.

    Returns:
        Sorted list of CIF file paths.
    """
    if args.single_cif is not None:
        return [args.single_cif]

    cif_files = sorted(args.input_dir.glob("*.cif"))
    if not cif_files:
        logger.warning("No .cif files found in %s", args.input_dir)
    return cif_files


def append_failure(failures_log: Path, structure_id: str, error: str) -> None:
    """Append a failure entry to the failures log file."""
    timestamp = datetime.datetime.now().isoformat()
    with open(failures_log, "a") as f:
        f.write(f"{timestamp}\t{structure_id}\t{error}\n")


def process_single_structure(
    cif_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    work_dir: Path,
    inputs_dir: Path,
    failures_log: Path,
    n_processed: int,
    n_failed: int,
    n_skipped: int,
) -> tuple[dict | None, int, int, int]:
    """Process a single CIF file through the full pipeline.

    Args:
        cif_path: Path to the CIF file.
        args: Parsed CLI arguments.
        output_dir: Directory for .pt output files.
        work_dir: Boltz working directory.
        inputs_dir: Directory for Boltz input YAML files.
        failures_log: Path to failures log file.
        n_processed: Current count of successfully processed structures.
        n_failed: Current count of failed structures.
        n_skipped: Current count of skipped structures.

    Returns:
        Tuple of (metrics_dict_or_None, n_processed, n_failed, n_skipped).
    """
    structure_id = cif_path.stem
    output_path = output_dir / f"{structure_id}.pt"

    # Check if already processed
    if output_path.exists() and not args.override:
        logger.info("[%s] Output already exists, skipping.", structure_id)
        n_skipped += 1
        return None, n_processed, n_failed, n_skipped

    # Step 1: Parse CIF chains
    try:
        chains: list[ChainInfo] = parse_cif_chains(cif_path)
    except ValueError as e:
        logger.warning("[%s] No protein chains found: %s", structure_id, e)
        n_skipped += 1
        return None, n_processed, n_failed, n_skipped
    except Exception as e:
        logger.warning("[%s] CIF parse failure: %s", structure_id, e)
        append_failure(failures_log, structure_id, f"CIF parse failure: {e}")
        n_failed += 1
        return None, n_processed, n_failed, n_skipped

    total_residues = sum(c.n_residues for c in chains)
    logger.info(
        "[%s] Parsed %d chain(s), %d total residues.",
        structure_id,
        len(chains),
        total_residues,
    )

    # Step 2: Generate Boltz YAML
    yaml_content = chains_to_boltz_yaml(chains, use_msa=args.use_msa_server)
    yaml_path = inputs_dir / f"{structure_id}.yaml"
    yaml_path.write_text(yaml_content)

    # Step 3: Run Boltz prediction
    t_start = time.time()
    result = run_boltz_predict(
        yaml_path=yaml_path,
        out_dir=work_dir,
        model=args.model,
        devices=args.devices,
        accelerator=args.accelerator,
        diffusion_samples=args.diffusion_samples,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
        use_msa_server=args.use_msa_server,
        override=args.override,
    )
    elapsed_s = time.time() - t_start

    if not result.success:
        logger.error("[%s] Boltz prediction failed: %s", structure_id, result.error_msg)
        append_failure(failures_log, structure_id, f"Boltz failure: {result.error_msg}")
        n_failed += 1
        return None, n_processed, n_failed, n_skipped

    if result.plddt is None:
        logger.error("[%s] Boltz returned no pLDDT data.", structure_id)
        append_failure(failures_log, structure_id, "Missing pLDDT data")
        n_failed += 1
        return None, n_processed, n_failed, n_skipped

    # Step 4: Verify shape
    plddt_np = result.plddt
    if plddt_np.shape[0] != total_residues:
        logger.warning(
            "[%s] pLDDT shape mismatch: got %d, expected %d residues. Saving anyway.",
            structure_id,
            plddt_np.shape[0],
            total_residues,
        )

    # Step 5: Convert to tensors and compute bins
    plddt_tensor = torch.tensor(plddt_np, dtype=torch.float32)
    plddt_bin_tensor = plddt_to_bin(plddt_tensor, num_bins=args.num_bins)

    # Step 6: Build and save .pt file
    sequences = {c.chain_id: c.sequence for c in chains}
    chain_lengths = {c.chain_id: c.n_residues for c in chains}

    label_data = {
        "structure_id": structure_id,
        "sequences": sequences,
        "plddt": plddt_tensor,
        "plddt_bin": plddt_bin_tensor,
        "chain_lengths": chain_lengths,
        "n_residues": total_residues,
    }
    torch.save(label_data, output_path)
    n_processed += 1

    logger.info(
        "[%s] Saved %s (pLDDT mean=%.3f, %d residues, %.1fs).",
        structure_id,
        output_path,
        float(plddt_tensor.mean()),
        plddt_np.shape[0],
        elapsed_s,
    )

    # Step 7: Log metrics to W&B
    metrics = log_protein_metrics(
        structure_id=structure_id,
        plddt=plddt_np,
        n_residues=total_residues,
        elapsed_s=elapsed_s,
        n_processed=n_processed,
        n_failed=n_failed,
        n_skipped=n_skipped,
    )

    return metrics, n_processed, n_failed, n_skipped


def main() -> None:
    """Main entry point for dataset generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Create directories
    output_dir = args.output_dir
    work_dir = args.work_dir
    inputs_dir = work_dir / "inputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    failures_log = work_dir / "failures.log"

    # Initialize W&B
    init_wandb_run(args)

    # Collect CIF files
    cif_files = collect_cif_files(args)
    logger.info("Found %d CIF file(s) to process.", len(cif_files))

    # Processing loop
    n_processed = 0
    n_failed = 0
    n_skipped = 0
    protein_stats: list[dict] = []

    for i, cif_path in enumerate(cif_files):
        structure_id = cif_path.stem
        logger.info(
            "Processing [%d/%d]: %s", i + 1, len(cif_files), structure_id
        )

        try:
            metrics, n_processed, n_failed, n_skipped = process_single_structure(
                cif_path=cif_path,
                args=args,
                output_dir=output_dir,
                work_dir=work_dir,
                inputs_dir=inputs_dir,
                failures_log=failures_log,
                n_processed=n_processed,
                n_failed=n_failed,
                n_skipped=n_skipped,
            )
            if metrics is not None:
                protein_stats.append(metrics)
        except Exception as e:
            logger.error("[%s] Unexpected error: %s", structure_id, e, exc_info=True)
            append_failure(failures_log, structure_id, f"Unexpected error: {e}")
            n_failed += 1

    # Summary
    log_dataset_summary(protein_stats)
    finish_wandb_run()

    logger.info(
        "Dataset generation complete. Processed: %d, Failed: %d, Skipped: %d, Total: %d",
        n_processed,
        n_failed,
        n_skipped,
        len(cif_files),
    )
    print(f"\n{'='*60}")
    print(f"Dataset Generation Summary")
    print(f"{'='*60}")
    print(f"  Total CIF files:  {len(cif_files)}")
    print(f"  Processed:        {n_processed}")
    print(f"  Failed:           {n_failed}")
    print(f"  Skipped:          {n_skipped}")
    print(f"  Output directory:  {output_dir.resolve()}")
    if n_failed > 0:
        print(f"  Failures log:      {failures_log.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
