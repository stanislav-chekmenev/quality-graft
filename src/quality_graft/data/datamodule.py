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

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from loguru import logger

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBLightningDataModule,
)
from src.la_proteina.proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from quality_graft.data.boltz_runner import run_boltz_predict_dir
from quality_graft.data.cif_utils import parse_cif_chains, chains_to_boltz_yaml
from quality_graft.data.plddt_utils import plddt_to_bin

PLDDT_STATUS_FILE = "plddt_status.csv"

# Attributes with [N, N, ...] shapes that DensePaddingDataLoader can't collate.
_UNCOLLATABLE_ATTRS = ("pde_logits",)


class _StripAttrsDataset(torch.utils.data.Dataset):
    """Wraps a dataset to remove attributes that can't be densely padded
    and optionally filter out samples missing plddt_logits."""

    def __init__(self, dataset, attrs=_UNCOLLATABLE_ATTRS, require_plddt_logits: bool = False):
        self.dataset = dataset
        self.attrs = attrs

        if require_plddt_logits:
            valid = []
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, "plddt_logits") and data.plddt_logits is not None:
                    valid.append(i)
            n_dropped = len(dataset) - len(valid)
            if n_dropped > 0:
                logger.warning(
                    "Dropped {}/{} samples missing plddt_logits.",
                    n_dropped, len(dataset),
                )
            self._valid_indices = valid
        else:
            self._valid_indices = None

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._valid_indices is not None:
            idx = self._valid_indices[idx]
        data = self.dataset[idx]
        for attr in self.attrs:
            if hasattr(data, attr):
                delattr(data, attr)
        return data


def _load_plddt_status(path: Path) -> Dict[str, bool]:
    """Load plddt_status.csv into {structure_id: has_plddt} dict."""
    status: Dict[str, bool] = {}
    if not path.exists():
        return status
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status[row["structure_id"]] = row["has_plddt"] == "true"
    return status


def _save_plddt_status(path: Path, status: Dict[str, bool]) -> None:
    """Write plddt_status.csv from {structure_id: has_plddt} dict."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "has_plddt"])
        writer.writeheader()
        for sid in sorted(status):
            writer.writerow({"structure_id": sid, "has_plddt": "true" if status[sid] else "false"})


def _get_plddt_set(status: Dict[str, bool]) -> Set[str]:
    """Return set of structure_ids that have pLDDT."""
    return {sid for sid, has in status.items() if has}


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
        local_only: bool = False,
        reprocess_boltz: bool = False,
        distillation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boltz_config = boltz_config
        self.num_plddt_bins = num_plddt_bins
        self.local_only = local_only
        self.reprocess_boltz = reprocess_boltz
        self.distillation = distillation
        self.boltz_work_dir = self.data_dir / "boltz_work"
        self.boltz_inputs_dir = self.boltz_work_dir / "inputs"

    @property
    def plddt_status_path(self) -> Path:
        return self.processed_dir / PLDDT_STATUS_FILE

    def setup(self, stage=None):
        if self.local_only:
            self._setup_local_only(stage)
        else:
            super().setup(stage)

        # Filter splits to only include structures that have pLDDT labels.
        if stage in ("fit", None) and self.dfs_splits is not None:
            plddt_set = _get_plddt_set(_load_plddt_status(self.plddt_status_path))

            for split_name in list(self.dfs_splits.keys()):
                df = self.dfs_splits[split_name]
                has_plddt = []
                for _, row in df.iterrows():
                    pdb = row["pdb"]
                    chain = row.get("chain")
                    sid = f"{pdb}_{chain}" if chain else pdb
                    has_plddt.append(sid in plddt_set)
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

        # Strip [N,N,...] attributes that DensePaddingDataLoader can't collate
        # and filter out samples missing plddt_logits when distilling
        if stage in ("fit", None):
            if self.train_ds is not None:
                self.train_ds = _StripAttrsDataset(
                    self.train_ds, require_plddt_logits=self.distillation,
                )
            if self.val_ds is not None:
                self.val_ds = _StripAttrsDataset(
                    self.val_ds, require_plddt_logits=self.distillation,
                )

    def _setup_local_only(self, stage=None):
        """Setup for local_only mode: build DataFrame from plddt_status.csv.

        Falls back to scanning .pt files if the CSV doesn't exist yet.
        """
        import pandas as pd

        plddt_status = _load_plddt_status(self.plddt_status_path)

        if plddt_status:
            # Fast path: use the CSV
            plddt_set = _get_plddt_set(plddt_status)
            records = []
            for sid in sorted(plddt_set):
                parts = sid.split("_", 1)
                pdb = parts[0]
                chain = parts[1] if len(parts) > 1 else None
                records.append({"pdb": pdb, "chain": chain, "id": sid})
            n_total = len(plddt_status)
            n_skipped = n_total - len(records)
        else:
            # Fallback: scan .pt files (first run before CSV exists)
            logger.warning(
                "plddt_status.csv not found, scanning .pt files (slow). "
                "Run preprocessing to generate it."
            )
            pt_files = sorted(self.processed_dir.glob("*.pt"))
            records = []
            n_skipped = 0
            status: Dict[str, bool] = {}
            for pt in pt_files:
                graph = torch.load(pt, weights_only=False)
                has = hasattr(graph, "plddt_bin") and graph.plddt_bin is not None
                status[pt.stem] = has
                if not has:
                    n_skipped += 1
                    continue
                parts = pt.stem.split("_", 1)
                pdb = parts[0]
                chain = parts[1] if len(parts) > 1 else None
                records.append({"pdb": pdb, "chain": chain, "id": pt.stem})
            n_total = len(pt_files)
            # Save the CSV so next time is fast
            _save_plddt_status(self.plddt_status_path, status)

        if not records:
            raise RuntimeError(
                f"local_only=True but 0 .pt files in "
                f"{self.processed_dir} have pLDDT labels. "
                "Run preprocessing (mode=preprocess) first."
            )
        self.df_data = pd.DataFrame(records)
        logger.info(
            "local_only: {} .pt files with pLDDT ({} skipped without labels, "
            "{} total)",
            len(records), n_skipped, n_total,
        )

        # Split using the existing datasplitter
        file_identifier = self.data_dir.name
        (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
            self.datasplitter.split_data(self.df_data, file_identifier)
        )

        if stage == "fit" or stage is None:
            self.train_ds = self._get_dataset("train")
            self.val_ds = self._get_dataset("val")
        elif stage == "test":
            self.test_ds = self._get_dataset("test")

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
        """Two-pass preprocessing: PyG conversion then Boltz-1 pLDDT labels.

        When ``reprocess_boltz=True``, skips Pass 1 (PDB download / PyG
        conversion) and re-runs Pass 2 (Boltz prediction) on **all**
        structures, overwriting existing pLDDT/logit data.
        """
        if self.local_only and not self.reprocess_boltz:
        """Two-pass preprocessing: PyG conversion then Boltz-1 pLDDT labels.

        When ``reprocess_boltz=True``, skips Pass 1 (PDB download / PyG
        conversion) and re-runs Pass 2 (Boltz prediction) on **all**
        structures, overwriting existing pLDDT/logit data.
        """
        if self.local_only and not self.reprocess_boltz:
            # Data already preprocessed — skip all preprocessing.
            if not self.processed_dir.exists() or not any(self.processed_dir.glob("*.pt")):
                raise RuntimeError(
                    f"local_only=True but no .pt files found in {self.processed_dir}. "
                    "Run preprocessing first."
                )
            logger.info(
                "local_only=True: skipping prepare_data entirely, "
                "{} .pt files in {}",
                len(list(self.processed_dir.glob("*.pt"))), self.processed_dir,
            )
            return

        if not self.reprocess_boltz:
            # Pass 1: parent handles filtering, download, PyG conversion
            super().prepare_data()
        if not self.reprocess_boltz:
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

        Sequential chunked pipeline:
          Phase 1: Prepare YAMLs into per-chunk subdirectories
          Phase 2: Process each chunk sequentially via run_boltz_predict_dir
                   with native multi-GPU (--devices N)
          Phase 3: Merge pLDDT into .pt files after each chunk, save CSV
        """
        chunk_size = self.boltz_config.get("chunk_size", 10)
        num_devices = self.boltz_config.get("num_devices", 1)

        # Phase 1: Clean stale chunk directories
        for stale in self.boltz_inputs_dir.glob("chunk_*"):
            if stale.is_dir():
                shutil.rmtree(stale)
        for stale in self.boltz_work_dir.glob("chunk_*"):
            if stale.is_dir():
                shutil.rmtree(stale)

        # Load pLDDT status from CSV (fast) instead of loading every .pt file
        plddt_status = _load_plddt_status(self.plddt_status_path)
        plddt_set = _get_plddt_set(plddt_status)

        submitted_ids: List[str] = []
        n_skipped = 0

        for fname in file_names:
            structure_id = fname.replace(".pt", "")
            if structure_id in plddt_set and not self.reprocess_boltz:
            if structure_id in plddt_set and not self.reprocess_boltz:
                n_skipped += 1
                continue
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
            "Splitting {} structures into {} chunks (chunk_size={}, devices={}).",
            len(submitted_ids), n_chunks, chunk_size, num_devices,
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
            "devices": num_devices,
            "accelerator": self.boltz_config.get("accelerator", "gpu"),
            "diffusion_samples": self.boltz_config.get("diffusion_samples", 1),
            "sampling_steps": self.boltz_config.get("sampling_steps", 200),
            "recycling_steps": self.boltz_config.get("recycling_steps", 3),
            "use_msa_server": self.boltz_config.get("use_msa_server", False),
            "timeout": chunk_timeout,
            "num_workers": self.boltz_config.get("num_workers", 2),
            "preprocessing_threads": self.boltz_config.get("preprocessing_threads"),
            "max_parallel_samples": self.boltz_config.get("max_parallel_samples"),
        }

        # Phase 2: Process chunks sequentially (each uses all GPUs via --devices)
        n_labeled = 0
        n_failed = 0

        for chunk_idx, (chunk_sids, inp_dir, out_dir) in enumerate(
            zip(valid_chunks, chunk_input_dirs, chunk_output_dirs)
        ):
            try:
                batch_result = run_boltz_predict_dir(
                    input_dir=inp_dir,
                    out_dir=out_dir,
                    structure_ids=chunk_sids,
                    **boltz_kwargs,
                )
            except Exception as e:
                logger.error("Chunk {} raised exception: {}", chunk_idx, e)
                n_failed += len(chunk_sids)
                for sid in chunk_sids:
                    plddt_status[sid] = False
                _save_plddt_status(self.plddt_status_path, plddt_status)
                continue

            # Check for OOM
            if batch_result.returncode != 0 and batch_result.error_msg:
                if "OOM" in batch_result.error_msg or "out of memory" in batch_result.error_msg.lower():
                    partial = len(batch_result.results)
                    logger.error(
                        "Chunk OOM: {}/{} structures completed before GPU memory exhaustion. "
                        "Will retry on re-run.",
                        partial, len(chunk_sids),
                    )

            # Phase 3: Merge pLDDT into .pt files for this chunk
            chunk_labeled = 0
            chunk_failed = 0

            for sid in chunk_sids:
                boltz_result = batch_result.results.get(sid)
                if boltz_result is None or boltz_result.plddt is None:
                    chunk_failed += 1
                    plddt_status[sid] = False
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
                    plddt_status[sid] = False
                    continue

                graph.plddt = torch.tensor(plddt_np, dtype=torch.float32)
                graph.plddt_bin = plddt_to_bin(graph.plddt, num_bins=self.num_plddt_bins)
                if boltz_result.plddt_logits is not None:
                    graph.plddt_logits = torch.tensor(
                        boltz_result.plddt_logits, dtype=torch.float32,
                    )
                if boltz_result.pde_logits is not None:
                    graph.pde_logits = torch.tensor(
                        boltz_result.pde_logits, dtype=torch.float32,
                    )
                if boltz_result.plddt_logits is not None:
                    graph.plddt_logits = torch.tensor(
                        boltz_result.plddt_logits, dtype=torch.float32,
                    )
                if boltz_result.pde_logits is not None:
                    graph.pde_logits = torch.tensor(
                        boltz_result.pde_logits, dtype=torch.float32,
                    )
                torch.save(graph, pt_path)
                chunk_labeled += 1
                plddt_status[sid] = True

            n_labeled += chunk_labeled
            n_failed += chunk_failed

            # Save status after each chunk so progress is preserved on crash
            _save_plddt_status(self.plddt_status_path, plddt_status)

            logger.info(
                "Chunks done: {}/{} | total labeled: {}/{} ({:.1f}%) | "
                "this chunk: {}/{} succeeded, {} failed",
                chunk_idx + 1, n_chunks,
                n_labeled, len(submitted_ids),
                100.0 * n_labeled / len(submitted_ids),
                chunk_labeled, len(chunk_sids), chunk_failed,
            )

        logger.info(
            "Boltz pass complete: {}/{} labeled, {} failed, {} skipped "
            "(already had pLDDT) | {} chunks, {} devices",
            n_labeled, len(submitted_ids), n_failed, n_skipped,
            n_chunks, num_devices,
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

        # Filter to the target chain so Boltz predicts single-chain pLDDT
        # matching the per-chain La-Proteina .pt files.
        parts = structure_id.split("_", 1)
        if len(parts) > 1:
            chain_id = parts[1]
            target = [c for c in chains if c.chain_id == chain_id]
            if not target:
                logger.warning(
                    "[{}] chain '{}' not found in CIF (available: {}), skipping.",
                    structure_id, chain_id,
                    [c.chain_id for c in chains],
                )
                return None
            chains = target

        use_msa = self.boltz_config.get("use_msa_server", False)
        yaml_content = chains_to_boltz_yaml(chains, use_msa=use_msa)
        yaml_path = output_dir / f"{structure_id}.yaml"
        yaml_path.write_text(yaml_content)
        return yaml_path
