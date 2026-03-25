#!/usr/bin/env python
"""Download UniProt reviewed (SwissProt) metadata TSV.

Fetches accession and length for all reviewed entries. Run once manually.

Usage:
    python scripts/download_uniprot_tsv.py --output metadata/swissprot/uniprot_metadata.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=tsv&query=(reviewed:true)&fields=accession,length"
)


def main():
    parser = argparse.ArgumentParser(description="Download UniProt SwissProt metadata TSV.")
    parser.add_argument(
        "--output",
        default="metadata/swissprot/uniprot_metadata.tsv",
        help="Output TSV path (default: metadata/swissprot/uniprot_metadata.tsv)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SwissProt metadata from UniProt...")
    print(f"URL: {UNIPROT_URL}")

    response = requests.get(UNIPROT_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(output, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Count lines (minus header)
    with open(output) as f:
        n_lines = sum(1 for _ in f) - 1
    print(f"Saved {n_lines} entries to {output}")


if __name__ == "__main__":
    main()
