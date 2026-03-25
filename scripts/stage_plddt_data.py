#!/usr/bin/env python3
"""Stage only .pt files that have pLDDT labels to a destination directory.

Reads plddt_status.csv from each source dir, copies only files marked
has_plddt=true, and writes a merged plddt_status.csv at the destination.

Usage:
    python scripts/stage_plddt_data.py \
        --src-dirs /path/to/dir1 /path/to/dir2 \
        --dest-dir /scratch/monomers_merged
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from loguru import logger


def load_plddt_ids(csv_path: Path) -> set[str]:
    """Return structure_ids where has_plddt=true."""
    ids = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["has_plddt"] == "true":
                ids.add(row["structure_id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Stage .pt files with pLDDT labels.")
    parser.add_argument("--src-dirs", nargs="+", required=True, help="Source data directories (each with processed/ subdir)")
    parser.add_argument("--dest-dir", required=True, help="Destination data directory")
    args = parser.parse_args()

    dest_processed = Path(args.dest_dir) / "processed"
    dest_processed.mkdir(parents=True, exist_ok=True)

    all_status: dict[str, bool] = {}
    total_copied = 0
    total_skipped = 0

    for src in args.src_dirs:
        src_processed = Path(src) / "processed"
        csv_path = src_processed / "plddt_status.csv"

        if not csv_path.exists():
            logger.warning("{} not found, skipping dir", csv_path)
            continue

        # Load full CSV for merging
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row["structure_id"]
                has = row["has_plddt"] == "true"
                # Last source wins on duplicates
                all_status[sid] = has

        plddt_ids = load_plddt_ids(csv_path)
        copied = 0
        missing = 0

        for sid in sorted(plddt_ids):
            src_pt = src_processed / f"{sid}.pt"
            dest_pt = dest_processed / f"{sid}.pt"

            if dest_pt.exists():
                total_skipped += 1
                continue

            if not src_pt.exists():
                missing += 1
                continue

            shutil.copy2(src_pt, dest_pt)
            copied += 1

        total_copied += copied
        logger.info("{}: copied {}, skipped {} existing, {} missing, {} with pLDDT",
                    Path(src).name, copied, total_skipped, missing, len(plddt_ids))

    # Write merged plddt_status.csv
    dest_csv = dest_processed / "plddt_status.csv"
    with open(dest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "has_plddt"])
        writer.writeheader()
        for sid in sorted(all_status):
            writer.writerow({"structure_id": sid, "has_plddt": "true" if all_status[sid] else "false"})

    n_plddt = sum(1 for v in all_status.values() if v)
    logger.info("Done: {} files copied, {} already present", total_copied, total_skipped)
    logger.info("Merged CSV: {} with pLDDT / {} total entries", n_plddt, len(all_status))


if __name__ == "__main__":
    main()
