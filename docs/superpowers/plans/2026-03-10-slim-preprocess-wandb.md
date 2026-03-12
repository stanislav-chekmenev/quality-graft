# Slim Preprocessing W&B Logger — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip W&B preprocessing logging down to a single end-of-run summary (histogram + 5 scalars) and fix loguru format strings so protein names/values render correctly.

**Architecture:** Remove per-protein `wandb.log()` from the hot loop entirely. Rewrite `wandb_logger.py` to only compute dataset-level stats from .pt files at the end. Fix all loguru calls from `%`-style to `{}`-style formatting.

**Tech Stack:** Python, W&B, loguru, numpy, torch, pytest

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/quality_graft/data/wandb_logger.py` | Rewrite | Dataset-level stats collection + single W&B summary log |
| `src/quality_graft/data/datamodule.py` | Modify | Remove per-protein W&B call, fix loguru format strings |
| `src/quality_graft/data/boltz_runner.py` | Modify | Fix loguru format strings |
| `scripts/train.py` | Modify | Update import (remove deleted function) |
| `tests/test_wandb_logger.py` | Rewrite | Tests for new slim wandb_logger |
| `tests/test_datamodule_wandb.py` | Rewrite | Tests for datamodule without per-protein logging |

---

### Task 1: Fix loguru format strings in datamodule.py and boltz_runner.py

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:92-165`
- Modify: `src/quality_graft/data/boltz_runner.py:206`

- [ ] **Step 1: Fix format strings in datamodule.py**

Change all `%d`/`%s`/`%.3f` to `{}`/`{:.3f}` in loguru logger calls:

```python
# Line 92 (was: "No .pt files found in %s, skipping Boltz pass.")
logger.warning("No .pt files found in {}, skipping Boltz pass.", self.processed_dir)

# Line 96 (was: "Starting Boltz-1 pLDDT pass on %d structures.")
logger.info("Starting Boltz-1 pLDDT pass on {} structures.", len(file_names))

# Lines 133-135 (was: "[%s] pLDDT length %d != graph residues %d, skipping.")
logger.warning(
    "[{}] pLDDT length {} != graph residues {}, skipping.",
    structure_id, plddt_np.shape[0], n_residues,
)

# Lines 146-149 (was: "[%s] pLDDT saved (mean=%.3f, %d residues).")
logger.info(
    "[{}] pLDDT saved (mean={:.3f}, {} residues).",
    structure_id, graph.plddt.mean().item(), n_residues,
)

# Lines 162-164 (was: "Boltz pass complete: processed=%d, skipped=%d, failed=%d")
logger.info(
    "Boltz pass complete: processed={}, skipped={}, failed={}",
    n_processed, n_skipped, n_failed,
)

# Lines 178 (was: "[%s] CIF not found: %s")
logger.warning("[{}] CIF not found: {}", structure_id, cif_path)

# Line 184 (was: "[%s] CIF parse failed: %s")
logger.warning("[{}] CIF parse failed: {}", structure_id, e)

# Line 208 (was: "[%s] Boltz failed: %s")
logger.warning("[{}] Boltz failed: {}", structure_id, result.error_msg)
```

- [ ] **Step 2: Fix format string in boltz_runner.py**

```python
# Line 206 (was: "Running Boltz: %s")
logger.info("Running Boltz: {}", " ".join(cmd))
```

- [ ] **Step 3: Run existing tests to confirm nothing breaks**

Run: `pytest tests/test_datamodule.py tests/test_datamodule_wandb.py -v`
Expected: All existing tests PASS (format string changes don't affect test behavior)

- [ ] **Step 4: Commit**

```bash
git add src/quality_graft/data/datamodule.py src/quality_graft/data/boltz_runner.py
git commit -m "fix: use loguru {} format strings instead of %s/%d"
```

---

### Task 2: Rewrite wandb_logger.py

**Files:**
- Rewrite: `src/quality_graft/data/wandb_logger.py`
- Rewrite: `tests/test_wandb_logger.py`

- [ ] **Step 1: Write failing tests for new wandb_logger**

Replace `tests/test_wandb_logger.py` entirely:

```python
"""Tests for slim wandb_logger dataset stats collection and summary logging."""

import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from torch_geometric.data import Data

from quality_graft.data.wandb_logger import collect_dataset_stats, log_dataset_summary


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Create a temp directory with fake .pt files containing pLDDT labels."""
    plddt_values = [
        torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),  # mean=0.7
        torch.tensor([0.95, 0.92, 0.88, 0.85, 0.80]),  # mean=0.88
        torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7]),  # mean=0.5
    ]
    for i, plddt in enumerate(plddt_values):
        graph = Data(
            coords=torch.randn(len(plddt), 3),
            plddt=plddt,
            plddt_bin=torch.zeros(len(plddt), dtype=torch.long),
        )
        torch.save(graph, tmp_path / f"structure_{i}.pt")

    # One file without pLDDT (should be skipped)
    graph_no_plddt = Data(coords=torch.randn(20, 3))
    torch.save(graph_no_plddt, tmp_path / "no_plddt.pt")

    return tmp_path


def test_collect_returns_only_labeled(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    assert len(stats) == 3


def test_collect_returns_tuples_with_correct_fields(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    for s in stats:
        assert "structure_id" in s
        assert "mean_plddt" in s
        assert "n_residues" in s
        assert isinstance(s["mean_plddt"], float)
        assert isinstance(s["n_residues"], int)


def test_collect_empty_dir(tmp_path):
    stats = collect_dataset_stats(tmp_path)
    assert stats == []


def test_collect_mean_plddt_values(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    means = sorted([s["mean_plddt"] for s in stats])
    assert abs(means[0] - 0.5) < 0.01
    assert abs(means[1] - 0.7) < 0.01
    assert abs(means[2] - 0.88) < 0.01


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_logs_correct_keys(mock_wandb):
    mock_wandb.run = MagicMock()
    mock_wandb.Histogram = MagicMock(return_value="histogram_obj")

    stats = [
        {"structure_id": "a", "mean_plddt": 0.7, "n_residues": 5},
        {"structure_id": "b", "mean_plddt": 0.88, "n_residues": 5},
        {"structure_id": "c", "mean_plddt": 0.5, "n_residues": 5},
    ]
    log_dataset_summary(stats)

    mock_wandb.log.assert_called_once()
    logged = mock_wandb.log.call_args[0][0]

    expected_keys = {
        "dataset/plddt_histogram",
        "dataset/mean_plddt",
        "dataset/std_plddt",
        "dataset/max_plddt",
        "dataset/min_plddt",
        "dataset/num_proteins",
    }
    assert set(logged.keys()) == expected_keys


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_scalar_values(mock_wandb):
    mock_wandb.run = MagicMock()
    mock_wandb.Histogram = MagicMock(return_value="histogram_obj")

    stats = [
        {"structure_id": "a", "mean_plddt": 0.6, "n_residues": 5},
        {"structure_id": "b", "mean_plddt": 0.8, "n_residues": 5},
    ]
    log_dataset_summary(stats)

    logged = mock_wandb.log.call_args[0][0]
    assert logged["dataset/num_proteins"] == 2
    assert abs(logged["dataset/mean_plddt"] - 0.7) < 1e-6
    assert abs(logged["dataset/max_plddt"] - 0.8) < 1e-6
    assert abs(logged["dataset/min_plddt"] - 0.6) < 1e-6


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_noop_when_no_wandb_run(mock_wandb):
    mock_wandb.run = None
    stats = [{"structure_id": "a", "mean_plddt": 0.7, "n_residues": 5}]
    log_dataset_summary(stats)
    mock_wandb.log.assert_not_called()


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_noop_when_empty_stats(mock_wandb):
    mock_wandb.run = MagicMock()
    log_dataset_summary([])
    mock_wandb.log.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_wandb_logger.py -v`
Expected: FAIL — old `collect_dataset_stats` returns dicts with `protein/` prefixed keys, old `log_dataset_summary` has different signature.

- [ ] **Step 3: Rewrite wandb_logger.py**

Replace `src/quality_graft/data/wandb_logger.py` entirely:

```python
"""W&B logging utilities for dataset preprocessing.

Logs a single dataset-level summary at the end of preprocessing:
histogram of per-protein mean pLDDT + scalar stats (mean, std, max, min, count).

All public functions are no-ops when W&B is not initialized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def collect_dataset_stats(processed_dir: Path) -> list[dict[str, Any]]:
    """Scan .pt files and collect per-protein mean pLDDT + residue count.

    Returns a list of dicts with keys: structure_id, mean_plddt, n_residues.
    Skips files without pLDDT labels.
    """
    import torch

    processed_dir = Path(processed_dir)
    pt_files = sorted(processed_dir.glob("*.pt"))
    stats: list[dict[str, Any]] = []

    for pt_path in pt_files:
        graph = torch.load(pt_path, weights_only=False)
        if not hasattr(graph, "plddt") or graph.plddt is None:
            continue

        plddt_np = graph.plddt.numpy()
        stats.append({
            "structure_id": pt_path.stem,
            "mean_plddt": float(plddt_np.mean()),
            "n_residues": int(plddt_np.shape[0]),
        })

    return stats


def log_dataset_summary(protein_stats: list[dict[str, Any]]) -> None:
    """Log dataset-level pLDDT summary to W&B.

    Logs: histogram of per-protein mean pLDDT, mean, std, max, min, count.
    No-op if wandb.run is None or stats is empty.
    """
    if wandb is None or wandb.run is None:
        return

    if not protein_stats:
        return

    means = np.array([s["mean_plddt"] for s in protein_stats])

    wandb.log({
        "dataset/plddt_histogram": wandb.Histogram(means),
        "dataset/mean_plddt": float(means.mean()),
        "dataset/std_plddt": float(means.std()),
        "dataset/max_plddt": float(means.max()),
        "dataset/min_plddt": float(means.min()),
        "dataset/num_proteins": len(protein_stats),
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wandb_logger.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/quality_graft/data/wandb_logger.py tests/test_wandb_logger.py
git commit -m "refactor: slim wandb_logger to dataset-level summary only"
```

---

### Task 3: Remove per-protein W&B logging from datamodule + update train.py

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:29,151-160`
- Modify: `scripts/train.py:42`
- Rewrite: `tests/test_datamodule_wandb.py`

- [ ] **Step 1: Write updated tests for datamodule (no per-protein logging)**

Replace `tests/test_datamodule_wandb.py` entirely:

```python
"""Tests verifying datamodule does NOT call per-protein W&B logging."""

from unittest.mock import patch, MagicMock

import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from pathlib import Path

from quality_graft.data.datamodule import QualityGraftDataModule


def _make_graph(n_residues=50, has_plddt=False):
    """Create a minimal PyG Data object."""
    graph = Data(coords=torch.randn(n_residues, 3))
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.zeros(n_residues, dtype=torch.long)
    return graph


@pytest.fixture
def mock_processed_dir(tmp_path):
    """Create processed dir with one unlabeled .pt file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    graph = _make_graph(n_residues=50, has_plddt=False)
    torch.save(graph, processed_dir / "test_structure.pt")
    return processed_dir


def test_run_boltz_pass_does_not_call_wandb(mock_processed_dir):
    """Verify _run_boltz_pass does NOT import or call wandb."""
    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = mock_processed_dir
    dm.num_plddt_bins = 50

    fake_plddt = np.random.rand(50).astype(np.float32)
    dm._run_boltz_for_structure = MagicMock(return_value=fake_plddt)

    with patch("quality_graft.data.datamodule.wandb", create=True) as mock_wandb:
        mock_wandb.run = None
        QualityGraftDataModule._run_boltz_pass(dm, ["test_structure.pt"])
        mock_wandb.log.assert_not_called()


def test_run_boltz_pass_saves_plddt(mock_processed_dir):
    """Verify pLDDT is saved correctly without W&B dependency."""
    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = mock_processed_dir
    dm.num_plddt_bins = 50

    fake_plddt = np.random.rand(50).astype(np.float32)
    dm._run_boltz_for_structure = MagicMock(return_value=fake_plddt)

    QualityGraftDataModule._run_boltz_pass(dm, ["test_structure.pt"])

    updated = torch.load(mock_processed_dir / "test_structure.pt", weights_only=False)
    assert hasattr(updated, "plddt")
    assert hasattr(updated, "plddt_bin")
    assert updated.plddt.shape[0] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datamodule_wandb.py -v`
Expected: FAIL — datamodule still imports and calls `log_protein_metrics`.

- [ ] **Step 3: Remove per-protein logging from datamodule.py**

In `src/quality_graft/data/datamodule.py`:

Remove the import on line 29:
```python
# DELETE this line:
from quality_graft.data.wandb_logger import log_protein_metrics
```

Remove the `log_protein_metrics()` call block (lines 151-160):
```python
# DELETE these lines from _run_boltz_pass:
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
```

Also remove the `time` import on line 15 and the timing variables (lines 122-123, 124 assignment of `elapsed_s`) since they're no longer needed:

Remove line 15: `import time`

In `_run_boltz_pass`, change:
```python
            t0 = time.time()
            plddt_np = self._run_boltz_for_structure(structure_id, pdb_code)
            elapsed_s = time.time() - t0
```
to:
```python
            plddt_np = self._run_boltz_for_structure(structure_id, pdb_code)
```

- [ ] **Step 4: Update train.py import**

In `scripts/train.py` line 42, the import already uses `collect_dataset_stats` and `log_dataset_summary` which still exist. No change needed — just verify the import still works.

Run: `python -c "from quality_graft.data.wandb_logger import collect_dataset_stats, log_dataset_summary; print('OK')"`

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_datamodule_wandb.py tests/test_datamodule.py tests/test_wandb_logger.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule_wandb.py
git commit -m "refactor: remove per-protein wandb logging from preprocessing loop"
```

---

### Task 4: Smoke test end-to-end

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (no regressions)

- [ ] **Step 2: Verify no stale imports**

Run: `python -c "from quality_graft.data.datamodule import QualityGraftDataModule; from quality_graft.data.wandb_logger import collect_dataset_stats, log_dataset_summary; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Commit (if any fixups needed)**

Only if fixes were required in previous steps.
