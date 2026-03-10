"""QualityGraftDataModule — extends PDBLightningDataModule with Boltz-1 pLDDT labels.

Two-pass preprocessing:
  Pass 1 (parent): PDB filtering, download, PyG conversion
  Pass 2 (this class): Boltz-1 prediction -> pLDDT labels merged into .pt files

Usage:
  dm = QualityGraftDataModule(data_dir="data/pdb/", boltz_config={...}, ...)
  dm.prepare_data()   # runs both passes
  dm.setup("fit")     # splits into train/val
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBLightningDataModule,
)
from src.la_proteina.proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from quality_graft.data.boltz_runner import run_boltz_predict
from quality_graft.data.cif_utils import parse_cif_chains, chains_to_boltz_yaml
from quality_graft.data.plddt_utils import plddt_to_bin

logger = logging.getLogger(__name__)


class QualityGraftDataModule(PDBLightningDataModule):
    """PDBLightningDataModule extended with Boltz-1 pLDDT label generation.

    After the parent class downloads and converts PDB structures to PyG
    Data objects, this class runs Boltz-1 predictions on each structure
    and stores pLDDT labels (continuous + binned) inside the .pt files.

    Parameters
    ----------
    boltz_config : dict
        Boltz prediction parameters: model, devices, accelerator,
        diffusion_samples, sampling_steps, recycling_steps, use_msa_server.
    num_plddt_bins : int
        Number of pLDDT bins (default 50, matching Boltz1 training).
    **kwargs
        All remaining arguments forwarded to PDBLightningDataModule.
    """

    def __init__(
        self,
        boltz_config: Dict[str, Any],
        num_plddt_bins: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boltz_config = boltz_config
        self.num_plddt_bins = num_plddt_bins
        self.boltz_work_dir = self.data_dir / "boltz_work"
        self.boltz_inputs_dir = self.boltz_work_dir / "inputs"

    def setup(self, stage=None):
        super().setup(stage)
        if stage in ("fit", None):
            if self.val_ds is not None and len(self.val_ds) == 0:
                logger.warning(
                    "Val split is empty. Using train split for validation (debug mode)."
                )
                self.val_ds = self.train_ds

    def val_dataloader(self):
        if self.val_ds is None:
            self.val_ds = self.val_dataset()
        return DensePaddingDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def prepare_data(self):
        """Two-pass preprocessing: PyG conversion then Boltz-1 pLDDT labels."""
        # Pass 1: parent handles filtering, download, PyG conversion
        super().prepare_data()

        # Pass 2: Boltz-1 pLDDT label generation
        pt_files = sorted(self.processed_dir.glob("*.pt"))
        if not pt_files:
            logger.warning("No .pt files found in %s, skipping Boltz pass.", self.processed_dir)
            return

        file_names = [f.name for f in pt_files]
        logger.info("Starting Boltz-1 pLDDT pass on %d structures.", len(file_names))

        self.boltz_work_dir.mkdir(parents=True, exist_ok=True)
        self.boltz_inputs_dir.mkdir(parents=True, exist_ok=True)

        self._run_boltz_pass(file_names)

    def _run_boltz_pass(self, file_names: List[str]) -> None:
        """Run Boltz-1 on structures that don't yet have pLDDT labels."""
        n_skipped = 0
        n_processed = 0
        n_failed = 0

        for fname in file_names:
            pt_path = self.processed_dir / fname
            graph = torch.load(pt_path, weights_only=False)

            # Skip if already has pLDDT labels
            if hasattr(graph, "plddt_bin") and graph.plddt_bin is not None:
                n_skipped += 1
                continue

            # Derive structure ID and find CIF
            structure_id = fname.replace(".pt", "")
            pdb_code = structure_id.split("_")[0]

            plddt_np = self._run_boltz_for_structure(structure_id, pdb_code)

            if plddt_np is None:
                n_failed += 1
                continue

            # Handle residue count mismatch
            n_residues = graph.coords.shape[0]
            if plddt_np.shape[0] != n_residues:
                logger.warning(
                    "[%s] pLDDT length %d != graph residues %d, skipping.",
                    structure_id, plddt_np.shape[0], n_residues,
                )
                n_failed += 1
                continue

            # Store labels in graph
            graph.plddt = torch.tensor(plddt_np, dtype=torch.float32)
            graph.plddt_bin = plddt_to_bin(graph.plddt, num_bins=self.num_plddt_bins)

            torch.save(graph, pt_path)
            n_processed += 1
            logger.info(
                "[%s] pLDDT saved (mean=%.3f, %d residues).",
                structure_id, graph.plddt.mean().item(), n_residues,
            )

        logger.info(
            "Boltz pass complete: processed=%d, skipped=%d, failed=%d",
            n_processed, n_skipped, n_failed,
        )

    def _run_boltz_for_structure(
        self, structure_id: str, pdb_code: str
    ) -> Optional[np.ndarray]:
        """Run Boltz-1 prediction for a single structure."""
        # Find CIF file
        cif_path = self.raw_dir / f"{pdb_code}.{self.format}"
        if not cif_path.exists():
            gz_path = cif_path.with_suffix(f".{self.format}.gz")
            if gz_path.exists():
                cif_path = gz_path
            else:
                logger.warning("[%s] CIF not found: %s", structure_id, cif_path)
                return None

        try:
            chains = parse_cif_chains(cif_path)
        except Exception as e:
            logger.warning("[%s] CIF parse failed: %s", structure_id, e)
            return None

        # Generate Boltz YAML
        use_msa = self.boltz_config.get("use_msa_server", False)
        yaml_content = chains_to_boltz_yaml(chains, use_msa=use_msa)
        yaml_path = self.boltz_inputs_dir / f"{structure_id}.yaml"
        yaml_path.write_text(yaml_content)

        # Run Boltz
        result = run_boltz_predict(
            yaml_path=yaml_path,
            out_dir=self.boltz_work_dir,
            model=self.boltz_config.get("model", "boltz1"),
            devices=self.boltz_config.get("devices", 1),
            accelerator=self.boltz_config.get("accelerator", "gpu"),
            diffusion_samples=self.boltz_config.get("diffusion_samples", 1),
            sampling_steps=self.boltz_config.get("sampling_steps", 200),
            recycling_steps=self.boltz_config.get("recycling_steps", 3),
            use_msa_server=use_msa,
            override=False,
        )

        if not result.success or result.plddt is None:
            logger.warning("[%s] Boltz failed: %s", structure_id, result.error_msg)
            return None

        return result.plddt
