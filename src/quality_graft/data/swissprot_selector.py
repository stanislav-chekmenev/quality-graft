"""SwissProtDataSelector — metadata-based filtering for AlphaFold SwissProt PDB files.

Unlike PDBDataSelector which queries RCSB via PDBManager, this selector works
entirely from a pre-downloaded UniProt metadata TSV and a directory of PDB files.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

import pandas as pd
from loguru import logger

from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSelector


class SwissProtDataSelector(PDBDataSelector):
    """Select AlphaFold SwissProt structures by metadata filtering + filesystem check.

    Parameters
    ----------
    source_dir : str
        Path to shared SwissProt PDB directory (e.g. /mnt/labs/shared/databases/swissprot_pdb_v4/files).
    metadata_tsv : str
        Path to UniProt TSV file with accession and length columns.
    alphafold_version : int
        AlphaFold model version for filename pattern (default 4).
    """

    def __init__(
        self,
        data_dir: str,
        source_dir: str,
        metadata_tsv: str,
        alphafold_version: int = 4,
        fraction: float = 1.0,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exclude_ids: Optional[List[str]] = None,
        exclude_ids_from_file: Optional[str] = None,
        num_workers: int = 32,
    ):
        super().__init__(
            data_dir=data_dir,
            fraction=fraction,
            min_length=min_length,
            max_length=max_length,
            exclude_ids=exclude_ids,
            exclude_ids_from_file=exclude_ids_from_file,
            num_workers=num_workers,
            molecule_type=None,
            experiment_types=None,
            oligomeric_min=None,
            oligomeric_max=None,
            best_resolution=None,
            worst_resolution=None,
            has_ligands=None,
            remove_ligands=None,
            remove_non_standard_residues=False,
            remove_pdb_unavailable=False,
            labels=None,
            remove_cath_unavailable=False,
        )
        self.database = "swissprot"
        self.source_dir = pathlib.Path(source_dir)
        self.metadata_tsv = pathlib.Path(metadata_tsv)
        self.alphafold_version = alphafold_version

    def create_dataset(self) -> pd.DataFrame:
        """Filter SwissProt structures by metadata and filesystem presence.

        Returns
        -------
        pd.DataFrame
            Columns: pdb, id, accession, length. No chain or sequence columns.
        """
        if self.df_data is not None:
            return self.df_data

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading UniProt metadata from {}", self.metadata_tsv)
        df = pd.read_csv(self.metadata_tsv, sep="\t")
        logger.info("Loaded {} entries from metadata TSV", len(df))
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"entry": "accession"})

        # Length filters
        if self.min_length is not None:
            df = df[df["length"] >= self.min_length]
            logger.info("{} entries after min_length={} filter", len(df), self.min_length)
        if self.max_length is not None:
            df = df[df["length"] <= self.max_length]
            logger.info("{} entries after max_length={} filter", len(df), self.max_length)

        # Fraction sampling
        if self.fraction < 1.0:
            df = df.sample(frac=self.fraction)
            logger.info("{} entries after fraction={} sampling", len(df), self.fraction)

        # Exclude IDs
        all_exclude = set()
        if self.exclude_ids:
            all_exclude.update(self.exclude_ids)
        if self.exclude_ids_from_file:
            with open(self.exclude_ids_from_file) as f:
                all_exclude.update(line.strip() for line in f if line.strip())
        if all_exclude:
            df = df[~df["accession"].isin(all_exclude)]
            logger.info("{} entries after excluding {} IDs", len(df), len(all_exclude))

        # Build expected filenames and cross-reference against source_dir
        v = self.alphafold_version
        df["pdb"] = df["accession"].apply(lambda acc: f"AF-{acc}-F1-model_v{v}")
        df["filename"] = df["pdb"] + ".pdb"

        existing_files = set(p.name for p in self.source_dir.iterdir() if p.is_file())
        df = df[df["filename"].isin(existing_files)]
        logger.info("{} entries after filesystem cross-reference", len(df))

        df["id"] = df["pdb"]
        self.df_data = df[["pdb", "id", "accession", "length"]].reset_index(drop=True)
        return self.df_data
