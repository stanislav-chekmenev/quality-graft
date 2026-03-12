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

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBLightningDataModule,
)
from src.la_proteina.proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from quality_graft.data.cif_utils import parse_cif_chains, chains_to_boltz_yaml
from quality_graft.data.plddt_utils import plddt_to_bin



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
            logger.warning("No .pt files found in {}, skipping Boltz pass.", self.processed_dir)
            return

        file_names = [f.name for f in pt_files]
        logger.info("Starting Boltz-1 pLDDT pass on {} structures.", len(file_names))

        self.boltz_work_dir.mkdir(parents=True, exist_ok=True)
        self.boltz_inputs_dir.mkdir(parents=True, exist_ok=True)

        self._run_boltz_pass(file_names)

    def _run_boltz_pass(self, file_names: List[str]) -> None:
        """Run Boltz-1 on structures that don't yet have pLDDT labels.

        Three-phase pipeline:
          Phase 1: Prepare YAMLs for structures needing pLDDT
          Phase 2: Single boltz predict invocation on the directory
          Phase 3: Collect results and merge into .pt files
        """
        # Phase 1: Prepare all YAMLs
        # Clear stale YAMLs from previous runs
        for old_yaml in self.boltz_inputs_dir.glob("*.yaml"):
            old_yaml.unlink()

        submitted_ids: List[str] = []
        n_skipped = 0

        for fname in file_names:
            pt_path = self.processed_dir / fname
            graph = torch.load(pt_path, weights_only=False)

            if hasattr(graph, "plddt_bin") and graph.plddt_bin is not None:
                n_skipped += 1
                continue

            structure_id = fname.replace(".pt", "")
            pdb_code = structure_id.split("_")[0]

            yaml_path = self._prepare_boltz_yaml(structure_id, pdb_code)
            if yaml_path is not None:
                submitted_ids.append(structure_id)

        logger.info(
            "Phase 1 complete: {} to process, {} skipped (already have pLDDT).",
            len(submitted_ids), n_skipped,
        )

        if not submitted_ids:
            logger.info("No structures need Boltz processing. Done.")
            return

        # Phase 2: Single Boltz invocation
        from quality_graft.data.boltz_runner import run_boltz_predict_dir

        batch_result = run_boltz_predict_dir(
            input_dir=self.boltz_inputs_dir,
            out_dir=self.boltz_work_dir,
            structure_ids=submitted_ids,
            model=self.boltz_config.get("model", "boltz1"),
            devices=self.boltz_config.get("devices", 1),
            accelerator=self.boltz_config.get("accelerator", "gpu"),
            diffusion_samples=self.boltz_config.get("diffusion_samples", 1),
            sampling_steps=self.boltz_config.get("sampling_steps", 200),
            recycling_steps=self.boltz_config.get("recycling_steps", 3),
            use_msa_server=self.boltz_config.get("use_msa_server", False),
        )

        # Phase 3: Collect results and merge into .pt files
        n_processed = 0
        n_failed = 0

        for structure_id in submitted_ids:
            fname = f"{structure_id}.pt"
            pt_path = self.processed_dir / fname
            graph = torch.load(pt_path, weights_only=False)

            boltz_result = batch_result.results.get(structure_id)
            if boltz_result is None or boltz_result.plddt is None:
                n_failed += 1
                continue

            plddt_np = boltz_result.plddt
            n_residues = graph.coords.shape[0]
            if plddt_np.shape[0] != n_residues:
                logger.warning(
                    "[{}] pLDDT length {} != graph residues {}, skipping.",
                    structure_id, plddt_np.shape[0], n_residues,
                )
                n_failed += 1
                continue

            graph.plddt = torch.tensor(plddt_np, dtype=torch.float32)
            graph.plddt_bin = plddt_to_bin(graph.plddt, num_bins=self.num_plddt_bins)

            torch.save(graph, pt_path)
            n_processed += 1
            logger.info(
                "[{}] pLDDT saved (mean={:.3f}, {} residues).",
                structure_id, graph.plddt.mean().item(), n_residues,
            )

        logger.info(
            "Boltz pass complete: processed={}, skipped={}, failed={}",
            n_processed, n_skipped, n_failed,
        )

    def _prepare_boltz_yaml(
        self, structure_id: str, pdb_code: str, output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Parse CIF and write Boltz input YAML. Returns yaml_path or None on failure."""
        if output_dir is None:
            output_dir = self.boltz_inputs_dir

        cif_path = self.raw_dir / f"{pdb_code}.{self.format}"
        if not cif_path.exists():
            gz_path = cif_path.with_suffix(f".{self.format}.gz")
            if gz_path.exists():
                cif_path = gz_path
            else:
                logger.warning("[{}] CIF not found: {}", structure_id, cif_path)
                return None

        try:
            chains = parse_cif_chains(cif_path)
        except Exception as e:
            logger.warning("[{}] CIF parse failed: {}", structure_id, e)
            return None

        use_msa = self.boltz_config.get("use_msa_server", False)
        yaml_content = chains_to_boltz_yaml(chains, use_msa=use_msa)
        yaml_path = output_dir / f"{structure_id}.yaml"
        yaml_path.write_text(yaml_content)
        return yaml_path
