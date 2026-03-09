# Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the full training pipeline — data module with integrated Boltz-1 pLDDT label generation, Lightning training module, metrics, and Hydra-configured train script.

**Architecture:** `QualityGraftDataModule` subclasses `PDBLightningDataModule` to add a second preprocessing pass that runs Boltz-1 on filtered structures and stores pLDDT labels in the PyG Data objects. `QualityGraftLightningModule` wraps the existing `QualityGraft(nn.Module)` with training/validation steps, masked cross-entropy loss, and four validation metrics. A Hydra-driven `scripts/train.py` supports `--mode=preprocess` and `--mode=train`.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, W&B, torch_geometric, existing vendored La-Proteina + Boltz code

---

## Task 1: Training Metrics

Standalone module with no dependencies on the data module or Lightning module.

**Files:**
- Create: `src/quality_graft/training/__init__.py`
- Create: `src/quality_graft/training/metrics.py`
- Create: `tests/test_metrics.py`

### Step 1: Create package init

Create `src/quality_graft/training/__init__.py`:

```python
"""Training utilities for Quality-Graft."""
```

### Step 2: Write failing tests for metrics

Create `tests/test_metrics.py`:

```python
"""Tests for pLDDT training metrics."""

import torch
import pytest


class TestPlddtAccuracy:
    """Tests for masked top-1 bin accuracy."""

    def test_perfect_prediction(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(2, 5, 50)
        labels = torch.tensor([[0, 1, 2, 3, 4], [10, 20, 30, 40, 49]])
        # Set the correct bin to have highest logit
        for b in range(2):
            for i in range(5):
                logits[b, i, labels[b, i]] = 10.0
        mask = torch.ones(2, 5)
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item() - 1.0) < 1e-6

    def test_zero_accuracy(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(1, 4, 50)
        labels = torch.tensor([[10, 20, 30, 40]])
        # Set wrong bins to highest
        for i in range(4):
            logits[0, i, (labels[0, i].item() + 1) % 50] = 10.0
        mask = torch.ones(1, 4)
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item()) < 1e-6

    def test_masking(self):
        from quality_graft.training.metrics import plddt_accuracy

        logits = torch.zeros(1, 4, 50)
        labels = torch.tensor([[0, 1, 2, 3]])
        # First two correct, last two wrong
        logits[0, 0, 0] = 10.0
        logits[0, 1, 1] = 10.0
        logits[0, 2, 49] = 10.0
        logits[0, 3, 49] = 10.0
        # Mask out the two wrong ones
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        acc = plddt_accuracy(logits, labels, mask)
        assert abs(acc.item() - 1.0) < 1e-6


class TestPlddtMae:
    """Tests for pLDDT mean absolute error."""

    def test_perfect_prediction(self):
        from quality_graft.training.metrics import plddt_mae

        # Logits that put all probability on the correct bin
        logits = torch.full((1, 3, 50), -100.0)
        labels = torch.tensor([[10, 20, 30]])
        for i in range(3):
            logits[0, i, labels[0, i]] = 100.0
        mask = torch.ones(1, 3)
        mae = plddt_mae(logits, labels, mask)
        assert mae.item() < 1e-4

    def test_masking(self):
        from quality_graft.training.metrics import plddt_mae

        logits = torch.full((1, 2, 50), -100.0)
        labels = torch.tensor([[10, 20]])
        # First residue correct, second totally wrong
        logits[0, 0, 10] = 100.0
        logits[0, 1, 0] = 100.0  # wrong bin
        # Mask out the wrong one
        mask = torch.tensor([[1.0, 0.0]])
        mae = plddt_mae(logits, labels, mask)
        assert mae.item() < 1e-4


class TestPearsonR:
    """Tests for per-protein Pearson correlation."""

    def test_perfect_correlation(self):
        from quality_graft.training.metrics import pearson_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        mask = torch.ones(1, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5

    def test_negative_correlation(self):
        from quality_graft.training.metrics import pearson_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.1]])
        mask = torch.ones(1, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item() - (-1.0)) < 1e-5

    def test_batch_averaging(self):
        from quality_graft.training.metrics import pearson_r

        # Two proteins: one perfect, one perfect negative
        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9],
                             [0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9],
                               [0.9, 0.7, 0.5, 0.3, 0.1]])
        mask = torch.ones(2, 5)
        r = pearson_r(pred, target, mask)
        assert abs(r.item()) < 1e-5  # average of 1.0 and -1.0


class TestSpearmanR:
    """Tests for per-protein Spearman rank correlation."""

    def test_perfect_rank_correlation(self):
        from quality_graft.training.metrics import spearman_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        target = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]])
        mask = torch.ones(1, 5)
        r = spearman_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5

    def test_masking(self):
        from quality_graft.training.metrics import spearman_r

        pred = torch.tensor([[0.1, 0.3, 0.5, 999.0, 999.0]])
        target = torch.tensor([[0.2, 0.4, 0.6, -999.0, -999.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        r = spearman_r(pred, target, mask)
        assert abs(r.item() - 1.0) < 1e-5
```

### Step 3: Run tests to verify they fail

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quality_graft.training'`

### Step 4: Implement metrics

Create `src/quality_graft/training/metrics.py`:

```python
"""Validation metrics for pLDDT prediction.

All metrics are computed per-protein (masked), then averaged across the batch.
"""

import torch
from torch import Tensor


def _bin_centers(num_bins: int = 50) -> Tensor:
    """Return bin center values in [0, 1] for the given number of bins."""
    bin_width = 1.0 / num_bins
    return torch.arange(num_bins, dtype=torch.float32) * bin_width + bin_width / 2


def _logits_to_continuous(logits: Tensor, num_bins: int = 50) -> Tensor:
    """Convert bin logits to continuous pLDDT via expected value.

    Args:
        logits: [b, n, num_bins]

    Returns:
        [b, n] continuous pLDDT in [0, 1]
    """
    probs = torch.softmax(logits, dim=-1)  # [b, n, num_bins]
    centers = _bin_centers(num_bins).to(probs.device, probs.dtype)  # [num_bins]
    return (probs * centers).sum(dim=-1)  # [b, n]


def _labels_to_continuous(labels: Tensor, num_bins: int = 50) -> Tensor:
    """Convert bin indices to continuous pLDDT via bin centers.

    Args:
        labels: [b, n] long tensor of bin indices

    Returns:
        [b, n] continuous pLDDT in [0, 1]
    """
    centers = _bin_centers(num_bins).to(labels.device)
    return centers[labels]


def plddt_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Masked top-1 bin prediction accuracy.

    Args:
        logits: [b, n, 50] predicted logits
        labels: [b, n] ground truth bin indices
        mask: [b, n] float mask (1 = valid, 0 = padding)

    Returns:
        Scalar accuracy averaged over all valid residues.
    """
    preds = logits.argmax(dim=-1)  # [b, n]
    correct = (preds == labels).float() * mask
    return correct.sum() / mask.sum().clamp(min=1)


def plddt_mae(logits: Tensor, labels: Tensor, mask: Tensor, num_bins: int = 50) -> Tensor:
    """Masked mean absolute error between predicted and ground truth pLDDT.

    Converts both to continuous [0, 1] via bin centers, then computes MAE.

    Args:
        logits: [b, n, num_bins] predicted logits
        labels: [b, n] ground truth bin indices
        mask: [b, n] float mask
        num_bins: number of pLDDT bins

    Returns:
        Scalar MAE averaged over all valid residues.
    """
    pred_continuous = _logits_to_continuous(logits, num_bins)
    target_continuous = _labels_to_continuous(labels, num_bins)
    ae = (pred_continuous - target_continuous).abs() * mask
    return ae.sum() / mask.sum().clamp(min=1)


def pearson_r(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Pearson correlation, averaged across the batch.

    Args:
        pred: [b, n] predicted continuous pLDDT
        target: [b, n] ground truth continuous pLDDT
        mask: [b, n] float mask

    Returns:
        Scalar mean Pearson r across proteins.
    """
    batch_size = pred.shape[0]
    rs = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred[i][m]
        t = target[i][m]
        if p.numel() < 3:
            continue
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        num = (p_centered * t_centered).sum()
        den = (p_centered.pow(2).sum() * t_centered.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(rs).mean()


def _rank(x: Tensor) -> Tensor:
    """Compute ranks (1-based) for a 1-D tensor. Ties get averaged rank."""
    sorted_indices = x.argsort()
    ranks = torch.empty_like(x)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    return ranks


def spearman_r(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Spearman rank correlation, averaged across the batch.

    Args:
        pred: [b, n] predicted continuous pLDDT
        target: [b, n] ground truth continuous pLDDT
        mask: [b, n] float mask

    Returns:
        Scalar mean Spearman r across proteins.
    """
    batch_size = pred.shape[0]
    rs = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred[i][m]
        t = target[i][m]
        if p.numel() < 3:
            continue
        p_ranked = _rank(p)
        t_ranked = _rank(t)
        p_centered = p_ranked - p_ranked.mean()
        t_centered = t_ranked - t_ranked.mean()
        num = (p_centered * t_centered).sum()
        den = (p_centered.pow(2).sum() * t_centered.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(rs).mean()
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/test_metrics.py -v`
Expected: All PASS

### Step 6: Commit

```bash
git add src/quality_graft/training/__init__.py src/quality_graft/training/metrics.py tests/test_metrics.py
git commit -m "feat: add pLDDT training metrics (accuracy, MAE, Pearson, Spearman)"
```

---

## Task 2: QualityGraftDataModule

**Files:**
- Create: `src/quality_graft/data/datamodule.py`
- Create: `tests/test_datamodule.py`

### Step 1: Write failing tests

Create `tests/test_datamodule.py`:

```python
"""Tests for QualityGraftDataModule.

Unit tests mock the Boltz runner to avoid GPU/network dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from quality_graft.data.datamodule import QualityGraftDataModule


def _make_fake_graph(pdb_id: str, n_residues: int = 10, has_plddt: bool = False) -> Data:
    """Create a minimal PyG Data object mimicking PDBDataset output."""
    graph = Data()
    graph.id = pdb_id
    graph.coords = torch.randn(n_residues, 37, 3)
    graph.coord_mask = torch.ones(n_residues, 37, dtype=torch.bool)
    graph.residue_type = torch.randint(0, 20, (n_residues,))
    graph.seq_pos = torch.arange(n_residues).unsqueeze(-1)
    graph.database = "pdb"
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.randint(0, 50, (n_residues,))
    return graph


class TestBoltzPassSkipsExisting:
    """Test that the Boltz pass skips graphs that already have pLDDT."""

    def test_skip_if_plddt_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()

            # Save a graph WITH plddt already
            graph = _make_fake_graph("test_pdb", has_plddt=True)
            torch.save(graph, processed / "test_pdb.pt")

            # Create a minimal CSV
            import pandas as pd
            df = pd.DataFrame({"pdb": ["test_pdb"], "id": ["test_pdb"]})
            df.to_csv(Path(tmpdir) / "test.csv", index=False)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )
            # _run_boltz_pass should detect existing plddt and skip
            with patch.object(dm, "_run_boltz_for_structure") as mock_boltz:
                dm._run_boltz_pass(["test_pdb.pt"])
                mock_boltz.assert_not_called()


class TestBoltzPassProcessesNew:
    """Test that the Boltz pass runs on graphs without pLDDT."""

    def test_processes_new_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()

            # Save a graph WITHOUT plddt
            graph = _make_fake_graph("test_pdb", n_residues=5, has_plddt=False)
            torch.save(graph, processed / "test_pdb.pt")

            # Create a dummy CIF file
            (raw / "test_pdb.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            with patch.object(dm, "_run_boltz_for_structure", return_value=fake_plddt):
                dm._run_boltz_pass(["test_pdb.pt"])

            # Reload and check
            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt")
            assert hasattr(updated, "plddt_bin")
            assert updated.plddt.shape[0] == 5
            assert updated.plddt_bin.shape[0] == 5
            assert updated.plddt_bin.dtype == torch.long
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_datamodule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quality_graft.data.datamodule'`

### Step 3: Implement QualityGraftDataModule

Create `src/quality_graft/data/datamodule.py`:

```python
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBDataSelector,
    PDBDataSplitter,
    PDBLightningDataModule,
)
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
        """Run Boltz-1 on structures that don't yet have pLDDT labels.

        Loads each .pt file, checks if plddt_bin already exists, and if
        not, runs Boltz-1 prediction and merges the labels.

        Parameters
        ----------
        file_names : list of str
            Filenames (e.g. "1abc_A.pt") in self.processed_dir.
        """
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
        """Run Boltz-1 prediction for a single structure.

        Finds the CIF file, generates a Boltz YAML input, runs prediction,
        and returns the per-residue pLDDT array.

        Parameters
        ----------
        structure_id : str
            Full structure ID (e.g. "1abc_A").
        pdb_code : str
            PDB code (e.g. "1abc") for finding the CIF file.

        Returns
        -------
        np.ndarray or None
            Per-residue pLDDT values [N] in [0, 1], or None on failure.
        """
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
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_datamodule.py -v`
Expected: All PASS

### Step 5: Commit

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule.py
git commit -m "feat: add QualityGraftDataModule with Boltz-1 pLDDT integration"
```

---

## Task 3: Lightning Training Module

**Files:**
- Create: `src/quality_graft/training/lightning_module.py`
- Create: `tests/test_lightning_module.py`

### Step 1: Write failing tests

Create `tests/test_lightning_module.py`:

```python
"""Tests for QualityGraftLightningModule.

Uses mock sub-modules (same as test_model_assembly.py) to avoid checkpoints.
"""

import pytest
import torch
import torch.nn as nn

from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.lightning_module import QualityGraftLightningModule

# Dimensions
TRUNK_DIM, PAIR_DIM, LATENT_DIM = 768, 256, 8
TARGET_S_DIM, TARGET_Z_DIM = 384, 128
B, N = 2, 10


class _MockLaProteinaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        b, n = batch["mask"].shape
        device = batch["mask"].device
        return {
            "trunk_seqs": torch.randn(b, n, TRUNK_DIM, device=device),
            "trunk_pair": torch.randn(b, n, n, PAIR_DIM, device=device),
            "local_latents": torch.randn(b, n, LATENT_DIM, device=device),
            "ca_coords": torch.randn(b, n, 3, device=device),
        }


class _MockConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self._s_to_plddt = nn.Linear(TARGET_S_DIM, 50, bias=False)
        self._s_to_resolved = nn.Linear(TARGET_S_DIM, 2, bias=False)
        self._z_to_pde = nn.Linear(TARGET_Z_DIM, 64, bias=False)
        self.requires_grad_(False)

    def forward(self, s, z, mask, use_kernels=False):
        return {
            "plddt_logits": self._s_to_plddt(s),
            "pde_logits": self._z_to_pde(z + z.transpose(1, 2)),
            "resolved_logits": self._s_to_resolved(s),
        }


def _make_module(n_attn_layers=0):
    model = QualityGraft(
        la_proteina=_MockLaProteinaWrapper(),
        adaptor=AdaptorModule(
            source_mode="trunk",
            trunk_dim=TRUNK_DIM, pair_dim=PAIR_DIM, latent_dim=LATENT_DIM,
            target_s_dim=TARGET_S_DIM, target_z_dim=TARGET_Z_DIM,
            n_attn_layers=n_attn_layers,
        ),
        confidence_head=_MockConfidenceHead(),
    )
    return QualityGraftLightningModule(
        model=model,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10,
        min_lr=1e-6,
        num_plddt_bins=50,
    )


def _make_batch():
    return {
        "coords_nm": torch.randn(B, N, 37, 3),
        "coord_mask": torch.ones(B, N, 37, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (B, N)),
        "mask": torch.ones(B, N, dtype=torch.float32),
        "plddt_bin": torch.randint(0, 50, (B, N)),
    }


class TestTrainingStep:
    def test_returns_scalar_loss(self):
        module = _make_module()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_is_finite(self):
        module = _make_module()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)


class TestValidationStep:
    def test_logs_all_metrics(self):
        module = _make_module()
        batch = _make_batch()
        # Capture logged metrics
        logged = {}
        module.log = lambda name, value, **kwargs: logged.update({name: value})
        module.validation_step(batch, batch_idx=0)
        expected_keys = {"val/loss", "val/plddt_accuracy", "val/plddt_mae",
                         "val/pearson_r", "val/spearman_r"}
        assert expected_keys.issubset(logged.keys())


class TestConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self):
        module = _make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_only_adaptor_params_in_optimizer(self):
        module = _make_module()
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        opt_params = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                opt_params.add(id(p))
        adaptor_params = {id(p) for p in module.model.adaptor.parameters()}
        assert opt_params == adaptor_params
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_lightning_module.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Implement Lightning module

Create `src/quality_graft/training/lightning_module.py`:

```python
"""PyTorch Lightning module for Quality-Graft training.

Wraps QualityGraft(nn.Module) with training/validation steps,
masked pLDDT cross-entropy loss, and validation metrics.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.metrics import (
    plddt_accuracy,
    plddt_mae,
    pearson_r,
    spearman_r,
    _logits_to_continuous,
    _labels_to_continuous,
)


class QualityGraftLightningModule(L.LightningModule):
    """Lightning wrapper for the Quality-Graft model.

    Parameters
    ----------
    model : QualityGraft
        The assembled model (La-Proteina + adaptor + confidence head).
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    betas : tuple[float, float]
        AdamW betas.
    warmup_steps : int
        Number of linear warmup steps.
    min_lr : float
        Minimum learning rate after linear decay.
    num_plddt_bins : int
        Number of pLDDT bins (default 50).
    """

    def __init__(
        self,
        model: QualityGraft,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 1e-6,
        num_plddt_bins: int = 50,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.num_plddt_bins = num_plddt_bins
        self.save_hyperparameters(ignore=["model"])

    def _compute_loss(self, plddt_logits: Tensor, plddt_labels: Tensor, mask: Tensor) -> Tensor:
        """Masked cross-entropy loss over pLDDT bins.

        Parameters
        ----------
        plddt_logits : [b, n, num_bins]
        plddt_labels : [b, n] long
        mask : [b, n] float (1=valid, 0=padding)
        """
        loss = F.cross_entropy(
            plddt_logits.reshape(-1, self.num_plddt_bins),
            plddt_labels.reshape(-1),
            reduction="none",
        )
        loss = loss.view_as(plddt_labels) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.float()
        loss = self._compute_loss(outputs["plddt_logits"], batch["plddt_bin"], mask)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.float()
        logits = outputs["plddt_logits"]
        labels = batch["plddt_bin"]

        # Loss
        loss = self._compute_loss(logits, labels, mask)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Metrics
        acc = plddt_accuracy(logits, labels, mask)
        mae = plddt_mae(logits, labels, mask, self.num_plddt_bins)

        pred_cont = _logits_to_continuous(logits, self.num_plddt_bins)
        target_cont = _labels_to_continuous(labels, self.num_plddt_bins)
        pr = pearson_r(pred_cont, target_cont, mask)
        sr = spearman_r(pred_cont, target_cont, mask)

        self.log("val/plddt_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log("val/plddt_mae", mae, sync_dist=True)
        self.log("val/pearson_r", pr, sync_dist=True)
        self.log("val/spearman_r", sr, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._lr_lambda
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _lr_lambda(self, step: int) -> float:
        """Linear warmup then linear decay to min_lr."""
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        # Linear decay from 1.0 to min_lr/lr over remaining training
        total_steps = self.trainer.estimated_stepping_batches
        decay_steps = total_steps - self.warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - self.warmup_steps) / decay_steps
        min_factor = self.min_lr / self.lr
        return max(1.0 - progress * (1.0 - min_factor), min_factor)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_lightning_module.py -v`
Expected: All PASS

### Step 5: Commit

```bash
git add src/quality_graft/training/lightning_module.py tests/test_lightning_module.py
git commit -m "feat: add QualityGraftLightningModule with pLDDT loss and metrics"
```

---

## Task 4: Hydra Config Updates

**Files:**
- Modify: `configs/training/default.yaml`
- Modify: `configs/data/dataset.yaml`
- Modify: `configs/data/preprocessing.yaml`

### Step 1: Update training config

Replace contents of `configs/training/default.yaml`:

```yaml
# Training configuration
optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.999]

scheduler:
  type: linear
  warmup_steps: 500
  min_lr: 1.0e-6

max_length: 128
batch_size: 4
num_workers: 4
precision: bf16
max_epochs: 50
gradient_clip_val: 1.0
accumulate_grad_batches: 1

wandb:
  project: quality-graft
  entity: null
  run_name: null
```

### Step 2: Update dataset config

Replace contents of `configs/data/dataset.yaml`:

```yaml
# Dataset configuration
data_dir: data/pdb/
max_length: ${training.max_length}
min_length: 10
molecule_type: protein
oligomeric_min: 1
oligomeric_max: 1
format: cif
num_plddt_bins: 50
batch_size: ${training.batch_size}
num_workers: ${training.num_workers}

boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 200
  recycling_steps: 3
  devices: 1
  accelerator: gpu
  use_msa_server: false
```

### Step 3: Update preprocessing config

Replace contents of `configs/data/preprocessing.yaml`:

```yaml
# Preprocessing is handled by QualityGraftDataModule.prepare_data()
# See configs/data/dataset.yaml for data module configuration.
```

### Step 4: Commit

```bash
git add configs/training/default.yaml configs/data/dataset.yaml configs/data/preprocessing.yaml
git commit -m "feat: populate Hydra configs for training pipeline"
```

---

## Task 5: Training Script

**Files:**
- Create: `scripts/train.py`
- Create: `tests/test_train_script.py`

### Step 1: Write failing test

Create `tests/test_train_script.py`:

```python
"""Smoke tests for scripts/train.py."""

import subprocess
import sys


class TestTrainScriptHelp:
    def test_help_flag(self):
        """train.py --help should exit cleanly."""
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "mode" in result.stdout.lower() or "usage" in result.stdout.lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_train_script.py -v`
Expected: FAIL — file not found or import error

### Step 3: Implement training script

Create `scripts/train.py`:

```python
#!/usr/bin/env python
"""Quality-Graft training script.

Usage:
    # Preprocess only (downloads, PyG conversion, Boltz-1 pLDDT labels)
    python scripts/train.py --mode=preprocess

    # Train (assumes preprocessing is done)
    python scripts/train.py --mode=train

    # Override config values
    python scripts/train.py --mode=train training.max_epochs=10 training.batch_size=2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# Ensure project paths are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBDataSelector,
    PDBDataSplitter,
)
from quality_graft.data.datamodule import QualityGraftDataModule
from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.confidence_head import BoltzConfidenceHead
from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper
from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.lightning_module import QualityGraftLightningModule

logger = logging.getLogger(__name__)


def build_data_module(cfg: DictConfig) -> QualityGraftDataModule:
    """Build the data module from Hydra config."""
    data_cfg = cfg.data.dataset

    dataselector = PDBDataSelector(
        data_dir=data_cfg.data_dir,
        max_length=data_cfg.max_length,
        min_length=data_cfg.min_length,
        molecule_type=data_cfg.molecule_type,
        oligomeric_min=data_cfg.oligomeric_min,
        oligomeric_max=data_cfg.oligomeric_max,
    )
    datasplitter = PDBDataSplitter(data_dir=data_cfg.data_dir)

    boltz_config = OmegaConf.to_container(data_cfg.boltz, resolve=True)

    return QualityGraftDataModule(
        data_dir=data_cfg.data_dir,
        dataselector=dataselector,
        datasplitter=datasplitter,
        format=data_cfg.format,
        boltz_config=boltz_config,
        num_plddt_bins=data_cfg.num_plddt_bins,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )


def build_model(cfg: DictConfig) -> QualityGraft:
    """Build the full QualityGraft model from Hydra config."""
    model_cfg = cfg.model

    # La-Proteina wrapper (from checkpoint)
    lp_cfg = model_cfg.la_proteina_wrapper
    la_proteina = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=lp_cfg.proteina_ckpt_path,
        autoencoder_ckpt_path=lp_cfg.autoencoder_ckpt_path,
        device=lp_cfg.device,
        use_decoder=lp_cfg.use_decoder,
        t_value=lp_cfg.t_value,
        deterministic_encode=lp_cfg.deterministic_encode,
    )

    # Adaptor (via Hydra instantiate)
    adaptor = hydra.utils.instantiate(model_cfg.quality_graft.adaptor)

    # Confidence head
    ch_cfg = model_cfg.quality_graft.confidence_head
    confidence_head = BoltzConfidenceHead(
        token_s=ch_cfg.token_s,
        token_z=ch_cfg.token_z,
        pairformer_args=OmegaConf.to_container(ch_cfg.pairformer_args, resolve=True),
        confidence_model_args=OmegaConf.to_container(ch_cfg.confidence_model_args, resolve=True),
        full_embedder_args=OmegaConf.to_container(ch_cfg.full_embedder_args, resolve=True),
        msa_args=OmegaConf.to_container(ch_cfg.msa_args, resolve=True),
        ckpt_path=ch_cfg.ckpt_path,
        ckpt_prefix=ch_cfg.ckpt_prefix,
        device=ch_cfg.device,
        freeze=ch_cfg.freeze,
        strict_loading=ch_cfg.strict_loading,
    )

    return QualityGraft(
        la_proteina=la_proteina,
        adaptor=adaptor,
        confidence_head=confidence_head,
    )


def build_lightning_module(cfg: DictConfig, model: QualityGraft) -> QualityGraftLightningModule:
    """Wrap the model in a Lightning module."""
    train_cfg = cfg.training

    return QualityGraftLightningModule(
        model=model,
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
        warmup_steps=train_cfg.scheduler.warmup_steps,
        min_lr=train_cfg.scheduler.min_lr,
        num_plddt_bins=cfg.data.dataset.num_plddt_bins,
    )


def build_trainer(cfg: DictConfig) -> L.Trainer:
    """Build the Lightning Trainer."""
    train_cfg = cfg.training

    # W&B logger
    wandb_logger = WandbLogger(
        project=train_cfg.wandb.project,
        entity=train_cfg.wandb.entity,
        name=train_cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    return L.Trainer(
        max_epochs=train_cfg.max_epochs,
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
    )


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    # Parse mode from sys.argv (before Hydra consumes args)
    mode = "train"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
            break

    logger.info("Mode: %s", mode)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if mode == "preprocess":
        dm = build_data_module(cfg)
        dm.prepare_data()
        logger.info("Preprocessing complete.")

    elif mode == "train":
        dm = build_data_module(cfg)
        dm.setup("fit")

        model = build_model(cfg)
        lit_module = build_lightning_module(cfg, model)
        trainer = build_trainer(cfg)

        logger.info(
            "Trainable params: %d, Frozen params: %d",
            model.num_trainable_parameters(),
            model.num_frozen_parameters(),
        )

        trainer.fit(lit_module, datamodule=dm)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'preprocess' or 'train'.")


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_train_script.py -v`
Expected: PASS

Note: The `--help` test may need adjustment depending on how Hydra handles `--help`. If Hydra intercepts it, test with `--cfg job` instead, or simply test that the module imports cleanly.

### Step 5: Commit

```bash
git add scripts/train.py tests/test_train_script.py
git commit -m "feat: add Hydra-driven training script with preprocess/train modes"
```

---

## Task 6: Run Existing Tests

Verify nothing is broken by running the full test suite.

### Step 1: Run all unit tests

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: All existing + new tests PASS

### Step 2: Fix any failures

If any tests fail, fix them before proceeding.

### Step 3: Commit any fixes

```bash
git add -u
git commit -m "fix: resolve test failures after training pipeline addition"
```

---

## Summary of Implementation Order

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Metrics | None |
| 2 | QualityGraftDataModule | Existing boltz_runner, cif_utils, plddt_utils |
| 3 | Lightning Module | Task 1 (metrics), existing QualityGraft |
| 4 | Hydra Configs | None |
| 5 | Training Script | Tasks 1-4 |
| 6 | Run All Tests | Tasks 1-5 |

Tasks 1, 2, and 4 are independent and can be parallelized.