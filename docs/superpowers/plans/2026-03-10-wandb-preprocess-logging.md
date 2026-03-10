# W&B Preprocess Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add W&B logging to the `mode=preprocess` path so that each Boltz-1 run logs per-protein metrics live, and a full dataset summary (plots + table) is logged after preprocessing completes.

**Architecture:** Hybrid approach — `train.py` owns the W&B run lifecycle (init/finish), `datamodule.py` calls per-protein logging during the Boltz pass (guarded by `wandb.run is not None`), and `train.py` triggers the full dataset summary after `prepare_data()` returns by scanning all labeled `.pt` files.

**Tech Stack:** wandb, torch, numpy, matplotlib, Hydra/OmegaConf

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/quality_graft/data/wandb_logger.py` | Modify | Add `collect_dataset_stats()`. Remove `init_wandb_run()` and `finish_wandb_run()`. |
| `src/quality_graft/data/datamodule.py` | Modify | Add timing + `log_protein_metrics()` calls in `_run_boltz_pass`. |
| `scripts/train.py` | Modify | Add W&B init/finish + summary logging in preprocess block. |
| `tests/test_wandb_logger.py` | Create | Unit tests for `collect_dataset_stats()` and updated logging integration. |

---

## Chunk 1: wandb_logger.py changes + tests

### Task 1: Add `collect_dataset_stats` to wandb_logger.py

**Files:**
- Modify: `src/quality_graft/data/wandb_logger.py:101-127` (remove `init_wandb_run`), lines `535-543` (remove `finish_wandb_run`)
- Modify: `src/quality_graft/data/wandb_logger.py` (add new function after `compute_protein_metrics`)
- Create: `tests/test_wandb_logger.py`

- [ ] **Step 1: Write failing test for `collect_dataset_stats`**

Create `tests/test_wandb_logger.py`:

```python
"""Tests for wandb_logger dataset stats collection."""

import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from torch_geometric.data import Data

from quality_graft.data.wandb_logger import collect_dataset_stats


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Create a temp directory with fake .pt files containing pLDDT labels."""
    for i in range(3):
        n_residues = 50 + i * 10
        plddt = torch.rand(n_residues)
        graph = Data(
            coords=torch.randn(n_residues, 3),
            plddt=plddt,
            plddt_bin=torch.zeros(n_residues, dtype=torch.long),
        )
        torch.save(graph, tmp_path / f"structure_{i}.pt")

    # One file without pLDDT (should be skipped)
    graph_no_plddt = Data(coords=torch.randn(20, 3))
    torch.save(graph_no_plddt, tmp_path / "no_plddt.pt")

    return tmp_path


def test_collect_dataset_stats_returns_all_labeled(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    assert len(stats) == 3  # only the 3 with pLDDT


def test_collect_dataset_stats_metric_keys(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    expected_keys = [
        "protein/structure_id",
        "protein/length",
        "protein/mean_plddt",
        "protein/median_plddt",
        "_plddt_array",
    ]
    for key in expected_keys:
        assert key in stats[0], f"Missing key: {key}"


def test_collect_dataset_stats_empty_dir(tmp_path):
    stats = collect_dataset_stats(tmp_path)
    assert stats == []


def test_collect_dataset_stats_plddt_array_is_numpy(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    for s in stats:
        assert isinstance(s["_plddt_array"], np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wandb_logger.py -v`
Expected: FAIL with `ImportError` or `cannot import name 'collect_dataset_stats'`

- [ ] **Step 3: Implement `collect_dataset_stats` in wandb_logger.py**

Add this function after the `compute_protein_metrics` function (after line 94) in `src/quality_graft/data/wandb_logger.py`:

```python
def collect_dataset_stats(processed_dir: "Path") -> list[dict[str, Any]]:
    """Scan all .pt files in processed_dir and collect metrics for those with pLDDT labels.

    Each labeled structure produces a metrics dict (from ``compute_protein_metrics``)
    plus a ``_plddt_array`` key holding the raw numpy array (needed by
    ``log_dataset_summary``).

    Parameters
    ----------
    processed_dir : Path
        Directory containing PyG ``.pt`` files with optional ``plddt`` attribute.

    Returns
    -------
    list[dict]
        One metrics dict per labeled structure.
    """
    from pathlib import Path as _Path
    import torch

    processed_dir = _Path(processed_dir)
    pt_files = sorted(processed_dir.glob("*.pt"))
    protein_stats: list[dict[str, Any]] = []

    for pt_path in pt_files:
        graph = torch.load(pt_path, weights_only=False)
        if not hasattr(graph, "plddt") or graph.plddt is None:
            continue

        structure_id = pt_path.stem
        plddt_np = graph.plddt.numpy()
        n_residues = plddt_np.shape[0]

        metrics = compute_protein_metrics(
            structure_id=structure_id,
            plddt=plddt_np,
            n_residues=n_residues,
            elapsed_s=0.0,
        )
        metrics["_plddt_array"] = plddt_np
        protein_stats.append(metrics)

    return protein_stats
```

Also add `from pathlib import Path` to the imports at the top of the file (it's not currently imported).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wandb_logger.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_wandb_logger.py src/quality_graft/data/wandb_logger.py
git commit -m "feat: add collect_dataset_stats to wandb_logger"
```

### Task 2: Remove `init_wandb_run` and `finish_wandb_run`

These functions are argparse-based and unused. `train.py` will call `wandb.init()` / `wandb.finish()` directly.

**Files:**
- Modify: `src/quality_graft/data/wandb_logger.py:101-127` (remove `init_wandb_run`)
- Modify: `src/quality_graft/data/wandb_logger.py:535-543` (remove `finish_wandb_run`)

- [ ] **Step 1: Verify no callers exist**

Run: `grep -r "init_wandb_run\|finish_wandb_run" src/ scripts/ tests/`
Expected: Only hits in `wandb_logger.py` itself (the definitions).

- [ ] **Step 2: Remove both functions**

Delete `init_wandb_run` (lines 101-127) and `finish_wandb_run` (lines 535-543) from `src/quality_graft/data/wandb_logger.py`. Also remove the `import argparse` at line 7 since it's no longer needed.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_wandb_logger.py -v`
Expected: All tests still PASS

- [ ] **Step 4: Commit**

```bash
git add src/quality_graft/data/wandb_logger.py
git commit -m "refactor: remove unused init_wandb_run and finish_wandb_run"
```

---

## Chunk 2: datamodule.py changes + tests

### Task 3: Add timing and per-protein W&B logging to `_run_boltz_pass`

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:102-151` (`_run_boltz_pass` method)
- Create: `tests/test_datamodule_wandb.py`

- [ ] **Step 1: Write failing test for per-protein logging during Boltz pass**

Create `tests/test_datamodule_wandb.py`:

```python
"""Tests for W&B logging integration in QualityGraftDataModule."""

import time
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from pathlib import Path


def _make_graph(n_residues=50, has_plddt=False):
    """Create a minimal PyG Data object."""
    graph = Data(coords=torch.randn(n_residues, 3))
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.zeros(n_residues, dtype=torch.long)
    return graph


@pytest.fixture
def mock_datamodule(tmp_path):
    """Create a minimal mock of QualityGraftDataModule with the methods we need."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    # Save a .pt file without pLDDT so it gets processed
    graph = _make_graph(n_residues=50, has_plddt=False)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    torch.save(graph, processed_dir / "test_structure.pt")

    # We can't easily instantiate the full datamodule (needs PDB dependencies),
    # so we test the _run_boltz_pass logic via patching
    return processed_dir


@patch("quality_graft.data.datamodule.log_protein_metrics")
def test_run_boltz_pass_calls_log_protein_metrics(mock_log, mock_datamodule):
    """Verify that _run_boltz_pass calls log_protein_metrics for successful structures."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    processed_dir = mock_datamodule

    # Create a mock datamodule instance with required attributes
    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = processed_dir
    dm.num_plddt_bins = 50

    # Mock _run_boltz_for_structure to return a valid pLDDT array
    fake_plddt = np.random.rand(50).astype(np.float32)
    dm._run_boltz_for_structure = MagicMock(return_value=fake_plddt)

    # Call the real method
    QualityGraftDataModule._run_boltz_pass(dm, ["test_structure.pt"])

    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args
    # Positional args: structure_id, plddt, n_residues, elapsed_s, n_processed, n_failed, n_skipped
    args = call_kwargs[1] if call_kwargs[1] else call_kwargs[0]
    if isinstance(args, tuple):
        assert args[0] == "test_structure"  # structure_id
        assert args[2] == 50  # n_residues


@patch("quality_graft.data.datamodule.log_protein_metrics")
def test_run_boltz_pass_skips_logging_for_already_labeled(mock_log, tmp_path):
    """Verify that already-labeled structures are skipped (no logging)."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    # Save a .pt file WITH pLDDT (already labeled)
    graph = _make_graph(n_residues=50, has_plddt=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    torch.save(graph, processed_dir / "labeled_structure.pt")

    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = processed_dir
    dm.num_plddt_bins = 50

    QualityGraftDataModule._run_boltz_pass(dm, ["labeled_structure.pt"])

    mock_log.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datamodule_wandb.py -v`
Expected: FAIL — `log_protein_metrics` is not yet imported in `datamodule.py`

- [ ] **Step 3: Modify `_run_boltz_pass` in datamodule.py**

In `src/quality_graft/data/datamodule.py`:

Add import at top (after existing imports from `quality_graft.data`):
```python
import time
from quality_graft.data.wandb_logger import log_protein_metrics
```

Replace the `_run_boltz_pass` method (lines 102-151) with:

```python
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

            t0 = time.time()
            plddt_np = self._run_boltz_for_structure(structure_id, pdb_code)
            elapsed_s = time.time() - t0

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

            # Log per-protein metrics to W&B (no-op if wandb.run is None)
            log_protein_metrics(
                structure_id=structure_id,
                plddt=plddt_np,
                n_residues=n_residues,
                elapsed_s=elapsed_s,
                n_processed=n_processed,
                n_failed=n_failed,
                n_skipped=n_skipped,
            )

        logger.info(
            "Boltz pass complete: processed=%d, skipped=%d, failed=%d",
            n_processed, n_skipped, n_failed,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datamodule_wandb.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing tests to check for regressions**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule_wandb.py
git commit -m "feat: add per-protein W&B logging to Boltz pass"
```

---

## Chunk 3: train.py preprocess orchestration

### Task 4: Add W&B lifecycle to preprocess mode in train.py

**Files:**
- Modify: `scripts/train.py:209-212` (preprocess block in `main()`)

- [ ] **Step 1: Modify the preprocess block in train.py**

In `scripts/train.py`, add an import near the top (after the existing wandb_logger-related imports area, around line 41):

```python
from quality_graft.data.wandb_logger import collect_dataset_stats, log_dataset_summary
```

Replace the preprocess block (lines 209-212) with:

```python
    if mode == "preprocess":
        # Init W&B run for preprocessing
        wandb_cfg = cfg.training.wandb
        try:
            import wandb

            wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=wandb_cfg.run_name,
                job_type="preprocessing",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning("W&B init failed, continuing without logging: %s", e)

        dm = build_data_module(cfg)
        dm.prepare_data()

        # Log full dataset summary (all labeled structures)
        try:
            protein_stats = collect_dataset_stats(dm.processed_dir)
            log_dataset_summary(protein_stats)
            logger.info(
                "Dataset summary logged: %d labeled structures.", len(protein_stats)
            )
        except Exception as e:
            logger.warning("Dataset summary logging failed: %s", e)

        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

        logger.info("Preprocessing complete.")
```

- [ ] **Step 2: Verify the script still parses**

Run: `python -c "import ast; ast.parse(open('scripts/train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add W&B run lifecycle to preprocess mode"
```

---

## Summary

| Task | Description | Files |
|---|---|---|
| 1 | Add `collect_dataset_stats` + tests | `wandb_logger.py`, `tests/test_wandb_logger.py` |
| 2 | Remove unused `init_wandb_run` / `finish_wandb_run` | `wandb_logger.py` |
| 3 | Add timing + per-protein W&B logging to Boltz pass | `datamodule.py`, `tests/test_datamodule_wandb.py` |
| 4 | Wire up W&B lifecycle in train.py preprocess mode | `train.py` |
