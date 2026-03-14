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

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBLightningDataModule,
)
from src.la_proteina.proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from quality_graft.data.boltz_runner import run_boltz_predict_dir
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

        # Filter splits to only include structures that have pLDDT labels.
        # Structures where Boltz failed/timed out during preprocessing will
        # have .pt files on disk but no plddt_bin attribute — these would
        # crash training_step.
        if stage in ("fit", None) and self.dfs_splits is not None:
            for split_name in list(self.dfs_splits.keys()):
                df = self.dfs_splits[split_name]
                has_plddt = []
                for _, row in df.iterrows():
                    pdb = row["pdb"]
                    chain = row.get("chain")
                    fname = f"{pdb}_{chain}.pt" if chain else f"{pdb}.pt"
                    pt_path = self.processed_dir / fname
                    if pt_path.exists():
                        graph = torch.load(pt_path, weights_only=False)
                        has_plddt.append(
                            hasattr(graph, "plddt_bin") and graph.plddt_bin is not None
                        )
                    else:
                        has_plddt.append(False)
                before = len(df)
                self.dfs_splits[split_name] = df[has_plddt].reset_index(drop=True)
                after = len(self.dfs_splits[split_name])
                if before != after:
                    logger.warning(
                        "Filtered {} split: {}/{} structures have pLDDT labels.",
                        split_name, after, before,
                    )

            # Rebuild datasets with filtered splits
            self.train_ds = self._get_dataset("train")
            self.val_ds = self._get_dataset("val")

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

        Parallel chunked pipeline:
          Phase 1: Prepare YAMLs into per-chunk subdirectories
          Phase 2: ThreadPoolExecutor submits run_boltz_predict_dir per chunk
          Phase 3: as_completed loop merges pLDDT into .pt files per chunk
        """
        num_boltz_workers = self.boltz_config.get("num_boltz_workers", 2)
        chunk_size = self.boltz_config.get("chunk_size", 10)

        # Phase 1: Clean stale chunk directories
        for stale in self.boltz_inputs_dir.glob("chunk_*"):
            if stale.is_dir():
                shutil.rmtree(stale)
        for stale in self.boltz_work_dir.glob("chunk_*"):
            if stale.is_dir():
                shutil.rmtree(stale)

        # Scan .pt files, skip those with pLDDT already
        submitted_ids: List[str] = []
        n_skipped = 0

        for fname in file_names:
            pt_path = self.processed_dir / fname
            graph = torch.load(pt_path, weights_only=False)

            if hasattr(graph, "plddt_bin") and graph.plddt_bin is not None:
                n_skipped += 1
                continue

            structure_id = fname.replace(".pt", "")
            submitted_ids.append(structure_id)

        logger.info(
            "Phase 1: {} to process, {} skipped (already have pLDDT).",
            len(submitted_ids), n_skipped,
        )

        if not submitted_ids:
            logger.info("No structures need Boltz processing. Done.")
            return

        # Split into chunks and prepare YAMLs into chunk directories
        chunks: List[List[str]] = []
        for i in range(0, len(submitted_ids), chunk_size):
            chunks.append(submitted_ids[i : i + chunk_size])

        n_chunks = len(chunks)
        logger.info(
            "Splitting {} structures into {} chunks (chunk_size={}, workers={}).",
            len(submitted_ids), n_chunks, chunk_size, num_boltz_workers,
        )

        # Prepare YAMLs into per-chunk input directories
        chunk_input_dirs: List[Path] = []
        chunk_output_dirs: List[Path] = []
        valid_chunks: List[List[str]] = []

        for chunk_idx, chunk_sids in enumerate(chunks):
            chunk_input_dir = self.boltz_inputs_dir / f"chunk_{chunk_idx:03d}"
            chunk_input_dir.mkdir(parents=True, exist_ok=True)
            chunk_output_dir = self.boltz_work_dir / f"chunk_{chunk_idx:03d}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            chunk_valid_sids = []
            for sid in chunk_sids:
                pdb_code = sid.split("_")[0]
                yaml_path = self._prepare_boltz_yaml(sid, pdb_code, output_dir=chunk_input_dir)
                if yaml_path is not None:
                    chunk_valid_sids.append(sid)

            if chunk_valid_sids:
                chunk_input_dirs.append(chunk_input_dir)
                chunk_output_dirs.append(chunk_output_dir)
                valid_chunks.append(chunk_valid_sids)

        n_chunks = len(valid_chunks)
        if n_chunks == 0:
            logger.warning("No valid YAMLs produced. Skipping Boltz.")
            return

        # Build boltz config kwargs (only keys accepted by run_boltz_predict_dir)
        timeout_per_structure = self.boltz_config.get("timeout_per_structure", 300)
        chunk_timeout = chunk_size * timeout_per_structure + 120  # +120s for model loading

        boltz_kwargs = {
            "model": self.boltz_config.get("model", "boltz1"),
            "devices": self.boltz_config.get("devices", 1),
            "accelerator": self.boltz_config.get("accelerator", "gpu"),
            "diffusion_samples": self.boltz_config.get("diffusion_samples", 1),
            "sampling_steps": self.boltz_config.get("sampling_steps", 200),
            "recycling_steps": self.boltz_config.get("recycling_steps", 3),
            "use_msa_server": self.boltz_config.get("use_msa_server", False),
            "timeout": chunk_timeout,
        }

        # Phase 2: Submit chunks to thread pool
        n_labeled = 0
        n_failed = 0
        chunks_done = 0

        with ThreadPoolExecutor(max_workers=num_boltz_workers) as executor:
            future_to_chunk = {}
            for idx, (chunk_sids, inp_dir, out_dir) in enumerate(
                zip(valid_chunks, chunk_input_dirs, chunk_output_dirs)
            ):
                future = executor.submit(
                    run_boltz_predict_dir,
                    input_dir=inp_dir,
                    out_dir=out_dir,
                    structure_ids=chunk_sids,
                    **boltz_kwargs,
                )
                future_to_chunk[future] = (idx, chunk_sids)

            # Phase 3: Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx, chunk_sids = future_to_chunk[future]
                chunks_done += 1

                try:
                    batch_result = future.result()
                except Exception as e:
                    logger.error("Chunk {} raised exception: {}", chunk_idx, e)
                    n_failed += len(chunk_sids)
                    continue

                # Check for OOM (boltz_runner formats OOM as "Boltz OOM: GPU memory exhaustion...")
                if batch_result.returncode != 0 and batch_result.error_msg:
                    if "OOM" in batch_result.error_msg or "out of memory" in batch_result.error_msg.lower():
                        partial = len(batch_result.results)
                        logger.error(
                            "Chunk OOM: {}/{} structures completed before GPU memory exhaustion. "
                            "Will retry on re-run.",
                            partial, len(chunk_sids),
                        )

                # Merge pLDDT into .pt files for this chunk
                chunk_labeled = 0
                chunk_failed = 0

                for sid in chunk_sids:
                    boltz_result = batch_result.results.get(sid)
                    if boltz_result is None or boltz_result.plddt is None:
                        chunk_failed += 1
                        continue

                    fname = f"{sid}.pt"
                    pt_path = self.processed_dir / fname
                    graph = torch.load(pt_path, weights_only=False)

                    plddt_np = boltz_result.plddt
                    n_residues = graph.coords.shape[0]
                    if plddt_np.shape[0] != n_residues:
                        logger.warning(
                            "[{}] pLDDT length {} != graph residues {}, skipping.",
                            sid, plddt_np.shape[0], n_residues,
                        )
                        chunk_failed += 1
                        continue

                    graph.plddt = torch.tensor(plddt_np, dtype=torch.float32)
                    graph.plddt_bin = plddt_to_bin(graph.plddt, num_bins=self.num_plddt_bins)
                    torch.save(graph, pt_path)
                    chunk_labeled += 1

                n_labeled += chunk_labeled
                n_failed += chunk_failed

                logger.info(
                    "Chunks done: {}/{} | total labeled: {}/{} ({:.1f}%) | "
                    "this chunk: {}/{} succeeded, {} failed",
                    chunks_done, n_chunks,
                    n_labeled, len(submitted_ids),
                    100.0 * n_labeled / len(submitted_ids),
                    chunk_labeled, len(chunk_sids), chunk_failed,
                )

        logger.info(
            "Boltz parallel pass complete: {}/{} labeled, {} failed, {} skipped "
            "(already had pLDDT) | {} chunks, {} workers",
            n_labeled, len(submitted_ids), n_failed, n_skipped,
            n_chunks, num_boltz_workers,
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
