"""SwissProtDataModule — single-pass processing for AlphaFold SwissProt structures.

Extracts pLDDT from B-factor column (0-100 scale) during PyG conversion.
No Boltz-1 prediction needed. No download step — files are pre-copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from loguru import logger

from src.la_proteina.openfold.np.residue_constants import resname_to_idx
from graphein.protein.tensor.io import protein_to_pyg
from quality_graft.data.datamodule import QualityGraftDataModule, _save_plddt_status
from quality_graft.data.plddt_utils import plddt_to_bin


class SwissProtDataModule(QualityGraftDataModule):
    """QualityGraftDataModule for AlphaFold SwissProt structures.

    Single-pass processing: pLDDT is extracted from B-factor during PyG
    conversion. No Boltz-1 prediction, no download step.

    Parameters
    ----------
    source_dir : str
        Path to shared SwissProt PDB directory.
    **kwargs
        All remaining arguments forwarded to QualityGraftDataModule.
    """

    def __init__(self, source_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.source_dir = Path(source_dir)

    def _get_file_identifier(self, ds):
        return f"df_swissprot_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}"

    def prepare_data(self):
        """Single-pass preprocessing: PyG conversion with pLDDT from B-factors.

        Does NOT call super().prepare_data() — that would trigger Boltz-1
        prediction from QualityGraftDataModule.
        """
        file_identifier = self._get_file_identifier(self.dataselector)
        df_data_name = f"{file_identifier}.csv"

        if not self.overwrite and (self.data_dir / df_data_name).exists():
            logger.info("{} already exists, skipping processing.", df_data_name)
            return

        df_data = self.dataselector.create_dataset()
        if len(df_data) == 0:
            raise ValueError(
                "SwissProtDataSelector returned 0 structures. "
                "Check metadata_tsv, source_dir, and filter parameters."
            )

        logger.info("Processing {} SwissProt structures.", len(df_data))

        # Process structures (chains=None for single-chain AlphaFold)
        self._process_structure_data(df_data["pdb"].tolist(), chains=None)

        # Save filtered DataFrame
        logger.info("Saving dataset CSV to {}", df_data_name)
        df_data.to_csv(self.data_dir / df_data_name, index=False)

        # Write plddt_status.csv from successfully created .pt files
        plddt_status = {}
        for pt_file in self.processed_dir.glob("*.pt"):
            plddt_status[pt_file.stem] = True

        _save_plddt_status(self.plddt_status_path, plddt_status)

        n_success = len(plddt_status)
        n_failed = len(df_data) - n_success
        logger.info(
            "SwissProt prepare_data complete: {} processed, {} failed.",
            n_success, n_failed,
        )

    def _load_and_process_pdb(
        self, index_pdb_tuple: Union[Tuple[int, str], Tuple[int, str, str]]
    ) -> Optional[str]:
        """Load PDB, convert to PyG graph, extract pLDDT from B-factor.

        Copies the parent method body from PDBLightningDataModule._load_and_process_pdb
        (pdb_data.py lines ~628-704) to avoid double I/O at 550K scale. The only
        additions are pLDDT extraction from B-factor and database tagging.

        If the parent method in pdb_data.py changes, this copy may silently diverge.
        """
        try:
            if len(index_pdb_tuple) == 3:
                i, pdb, chains = index_pdb_tuple
            elif len(index_pdb_tuple) == 2:
                i, pdb = index_pdb_tuple
                chains = "all"
            else:
                raise ValueError("index_pdb_tuple must have 2 or 3 elements")

            path = self.raw_dir / f"{pdb}.{self.format}"
            if path.exists():
                path = str(path)
            elif path.with_suffix("." + self.format + ".gz").exists():
                path = str(path.with_suffix("." + self.format + ".gz"))
            else:
                raise FileNotFoundError(
                    f"{pdb} not found in raw directory. "
                    f"Are you sure it's downloaded and has the format {self.format}?"
                )

            fill_value_coords = 1e-5
            graph = protein_to_pyg(
                path=path,
                chain_selection=chains,
                keep_insertions=True,
                store_het=self.store_het,
                store_bfactor=self.store_bfactor,
                fill_value_coords=fill_value_coords,
            )

        except Exception as e:
            logger.warning("Error processing {} {}: {}", pdb, chains, e)
            return None

        fname = f"{pdb}.pt" if chains == "all" else f"{pdb}_{chains}.pt"

        graph.id = fname.split(".")[0]
        coord_mask = graph.coords != fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        graph.residue_type = torch.tensor(
            [resname_to_idx[residue] for residue in graph.residues]
        ).long()
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

        # --- SwissProt additions: pLDDT from B-factor ---
        # graphein already averages bfactor per residue → bfactor is 1D [n_residues]
        # Use bfactor directly (not bfactor_avg which is a scalar from mean of 1D)
        graph.plddt = graph.bfactor / 100.0            # B-factor is pLDDT on 0-100 scale
        graph.plddt_bin = plddt_to_bin(graph.plddt)    # bin to 0..49
        graph.plddt_logits = None                      # hard targets only
        graph.database = "swissprot"

        if self.pre_transform:
            graph = self.pre_transform(graph)

        if self.pre_filter:
            if self.pre_filter(graph) is not True:
                return None

        torch.save(graph, self.processed_dir / fname)
        return fname
