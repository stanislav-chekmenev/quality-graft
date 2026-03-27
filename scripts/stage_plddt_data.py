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

    # 1. Scan destination for already-present files
    existing_ids = {p.stem for p in dest_processed.glob("*.pt")}
    logger.info("Destination already has {} .pt files", len(existing_ids))

    all_status: dict[str, bool] = {}
    total_copied = 0

    # 2. Walk source dirs; copy files with pLDDT, overwrite stale dest files
    #    (dest file is "stale" if source has pLDDT but dest is smaller,
    #     meaning it lacks pLDDT tensors added by the Boltz pass)
    for src in args.src_dirs:
        src_processed = Path(src) / "processed"
        csv_path = src_processed / "plddt_status.csv"

        if not csv_path.exists():
            logger.warning("{} not found, skipping dir", csv_path)
            continue

        # Load full CSV for merging (true wins over false for same sid)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row["structure_id"]
                has = row["has_plddt"] == "true"
                all_status[sid] = all_status.get(sid, False) or has

        plddt_ids = load_plddt_ids(csv_path)
        copied = 0
        replaced = 0
        skipped = 0
        missing = 0

        for sid in sorted(plddt_ids):
            src_pt = src_processed / f"{sid}.pt"
            if not src_pt.exists():
                missing += 1
                continue

            dest_pt = dest_processed / f"{sid}.pt"
            if dest_pt.exists():
                # Skip only if dest file has the exact same size as source.
                # Previously used >=, but that could skip stale files that
                # are large yet missing fields like plddt_logits.
                if dest_pt.stat().st_size == src_pt.stat().st_size:
                    skipped += 1
                    continue
                replaced += 1

            shutil.copy2(src_pt, dest_pt)
            copied += 1

        total_copied += copied
        existing_ids.update(plddt_ids)
        logger.info("{}: copied {} ({} replaced stale), skipped {} up-to-date, "
                    "{} missing source, {} with pLDDT",
                    Path(src).name, copied, replaced, skipped, missing, len(plddt_ids))

    # Write merged plddt_status.csv
    dest_csv = dest_processed / "plddt_status.csv"
    with open(dest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "has_plddt"])
        writer.writeheader()
        for sid in sorted(all_status):
            writer.writerow({"structure_id": sid, "has_plddt": "true" if all_status[sid] else "false"})

    n_plddt = sum(1 for v in all_status.values() if v)
    logger.info("Done: {} files copied, {} already present at start", total_copied, len(existing_ids) - total_copied)
    logger.info("Merged CSV: {} with pLDDT / {} total entries", n_plddt, len(all_status))


if __name__ == "__main__":
    main()
