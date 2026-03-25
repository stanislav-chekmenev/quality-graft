#!/usr/bin/env python
"""Copy filtered SwissProt PDB files from shared storage to scratch.

Idempotent: re-running only copies files that don't exist in dest-dir.

Usage:
    python scripts/copy_swissprot.py \
      --source-dir /mnt/labs/shared/databases/swissprot_pdb_v4/files \
      --dest-dir /scratch/schekmenev/swissprot_v4/raw \
      --metadata-tsv data/metadata/swissprot/uniprot_metadata.tsv \
      --min-length 30 \
      --max-length 512
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stdout, level="INFO")

# Ensure project paths are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LA_PROTEINA_DIR = SRC_DIR / "la_proteina"
for p in [PROJECT_ROOT, SRC_DIR, LA_PROTEINA_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quality_graft.data.swissprot_selector import SwissProtDataSelector


def main():
    parser = argparse.ArgumentParser(description="Copy filtered SwissProt PDB files to scratch.")
    parser.add_argument("--source-dir", required=True, help="Source SwissProt PDB directory")
    parser.add_argument("--dest-dir", required=True, help="Destination raw/ directory")
    parser.add_argument("--metadata-tsv", required=True, help="UniProt metadata TSV path")
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exclude-ids-file", default=None)
    parser.add_argument("--alphafold-version", type=int, default=4)
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary data_dir (selector needs it but we don't use it)
    selector = SwissProtDataSelector(
        data_dir=str(dest_dir.parent),
        source_dir=args.source_dir,
        metadata_tsv=args.metadata_tsv,
        alphafold_version=args.alphafold_version,
        fraction=args.fraction,
        min_length=args.min_length,
        max_length=args.max_length,
        exclude_ids_from_file=args.exclude_ids_file,
    )

    logger.info("Running SwissProtDataSelector.create_dataset()...")
    df = selector.create_dataset()
    logger.info(f"Filtered: {len(df)} structures")

    source_dir = Path(args.source_dir)
    logger.info("Scanning destination for existing files...")
    existing = set(p.name for p in dest_dir.iterdir() if p.is_file())

    logger.info("Found {} existing files in destination".format(len(existing)))
    logger.info("Determining files to copy...")    
    all_fnames = [f"{pdb}.pdb" for pdb in df["pdb"]]
    to_copy = [f for f in all_fnames if f not in existing]

    logger.info(f"Already present: {len(df) - len(to_copy)}")
    logger.info(f"To copy: {len(to_copy)}")

    for fname in tqdm(to_copy, desc="Copying"):
        shutil.copy2(source_dir / fname, dest_dir / fname)

    # Save filtered file list
    ids_path = dest_dir.parent / "filtered_ids.txt"
    ids_path.write_text("\n".join(df["accession"].tolist()) + "\n")
    logger.info(f"Filtered IDs saved to {ids_path}")
    logger.info(f"Done: {len(to_copy)} copied, {len(df)} total filtered")


if __name__ == "__main__":
    main()
