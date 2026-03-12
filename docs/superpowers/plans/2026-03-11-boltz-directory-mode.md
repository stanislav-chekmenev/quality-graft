# Boltz Directory Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-structure Boltz model loading by passing a directory of YAMLs to a single `boltz predict` invocation.

**Architecture:** Restructure `_run_boltz_pass()` into three phases (prepare YAMLs → single Boltz subprocess → collect results). Add `BoltzBatchResult` dataclass and `run_boltz_predict_dir()` to `boltz_runner.py`. OOM detection via stderr inspection with graceful partial-result collection.

**Tech Stack:** Python, subprocess, torch, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-boltz-directory-mode-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/quality_graft/data/boltz_runner.py` | Modify | Add `BoltzBatchResult` dataclass + `run_boltz_predict_dir()` function |
| `src/quality_graft/data/datamodule.py` | Modify | Restructure `_run_boltz_pass()` into 3 phases, remove `_run_boltz_for_structure()` |
| `tests/test_boltz_runner.py` | Create | Unit tests for `run_boltz_predict_dir()` and OOM detection |
| `tests/test_datamodule.py` | Modify | Update mocks from `_run_boltz_for_structure` to `run_boltz_predict_dir` |

---

## Chunk 1: `run_boltz_predict_dir()` in boltz_runner.py

### Task 1: Add `BoltzBatchResult` dataclass

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py:49-57` (after existing `BoltzResult`)
- Test: `tests/test_boltz_runner.py` (create)

- [ ] **Step 1: Write the test for BoltzBatchResult**

Create `tests/test_boltz_runner.py`:

```python
"""Tests for boltz_runner batch functionality."""

from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult


class TestBoltzBatchResult:
    """Test BoltzBatchResult dataclass."""

    def test_construction(self):
        result = BoltzBatchResult(
            results={},
            n_submitted=5,
            returncode=0,
            error_msg=None,
        )
        assert result.n_submitted == 5
        assert result.returncode == 0
        assert result.results == {}
        assert result.error_msg is None

    def test_with_results(self):
        import numpy as np
        br = BoltzResult(
            structure_id="1ubq_A",
            plddt=np.array([0.8, 0.9]),
            confidence_json=None,
            success=True,
            error_msg=None,
        )
        result = BoltzBatchResult(
            results={"1ubq_A": br},
            n_submitted=3,
            returncode=0,
            error_msg=None,
        )
        assert len(result.results) == 1
        assert result.results["1ubq_A"].success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boltz_runner.py::TestBoltzBatchResult -v`
Expected: FAIL with `ImportError: cannot import name 'BoltzBatchResult'`

- [ ] **Step 3: Implement `BoltzBatchResult`**

In `src/quality_graft/data/boltz_runner.py`, add after the `BoltzResult` dataclass (after line 57):

```python
@dataclass
class BoltzBatchResult:
    """Result of a batch Boltz prediction run on a directory of YAMLs."""

    results: dict[str, BoltzResult]  # structure_id -> result, for outputs found
    n_submitted: int  # number of YAMLs in the input directory
    returncode: int  # subprocess exit code
    error_msg: str | None  # stderr summary if non-zero exit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boltz_runner.py::TestBoltzBatchResult -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_boltz_runner.py src/quality_graft/data/boltz_runner.py
git commit -m "Add BoltzBatchResult dataclass for directory-mode predictions"
```

---

### Task 2: Add `run_boltz_predict_dir()` — subprocess + result collection

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py` (append new function)
- Test: `tests/test_boltz_runner.py` (extend)

- [ ] **Step 1: Write the test for successful directory prediction**

Append to `tests/test_boltz_runner.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import numpy as np

from quality_graft.data.boltz_runner import run_boltz_predict_dir, BoltzBatchResult


class TestRunBoltzPredictDir:
    """Test run_boltz_predict_dir()."""

    def test_successful_batch(self, tmp_path):
        """All structures processed successfully."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        # Write two dummy YAML files
        (input_dir / "1ubq_A.yaml").write_text("dummy")
        (input_dir / "2abc_B.yaml").write_text("dummy")

        structure_ids = ["1ubq_A", "2abc_B"]

        # Create fake pLDDT output files where find_plddt_npz will look
        for sid in structure_ids:
            pred_dir = out_dir / "predictions" / sid
            pred_dir.mkdir(parents=True)
            np.savez(pred_dir / f"plddt_{sid}_model_0.npz", plddt=np.array([0.8, 0.9]))

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert isinstance(result, BoltzBatchResult)
        assert result.n_submitted == 2
        assert result.returncode == 0
        assert result.error_msg is None
        assert len(result.results) == 2
        assert result.results["1ubq_A"].success is True
        assert result.results["2abc_B"].success is True
        np.testing.assert_array_equal(result.results["1ubq_A"].plddt, [0.8, 0.9])

    def test_partial_results_on_crash(self, tmp_path):
        """Boltz crashes mid-run, but some outputs exist."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq_A.yaml").write_text("dummy")
        (input_dir / "2abc_B.yaml").write_text("dummy")

        structure_ids = ["1ubq_A", "2abc_B"]

        # Only first structure has output
        pred_dir = out_dir / "predictions" / "1ubq_A"
        pred_dir.mkdir(parents=True)
        np.savez(pred_dir / "plddt_1ubq_A_model_0.npz", plddt=np.array([0.7]))

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Some error"
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert result.returncode == 1
        assert result.error_msg is not None
        assert len(result.results) == 1
        assert "1ubq_A" in result.results
        assert "2abc_B" not in result.results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boltz_runner.py::TestRunBoltzPredictDir -v`
Expected: FAIL with `ImportError: cannot import name 'run_boltz_predict_dir'`

- [ ] **Step 3: Implement `run_boltz_predict_dir()`**

Append to `src/quality_graft/data/boltz_runner.py`:

```python
def run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    structure_ids: list[str],
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
) -> BoltzBatchResult:
    """Run boltz predict on a directory of YAMLs and collect results.

    Passes the entire input_dir to a single `boltz predict` invocation.
    After the subprocess finishes (or crashes), iterates over structure_ids
    and collects whatever pLDDT outputs exist.

    Args:
        input_dir: Directory containing YAML files for Boltz.
        out_dir: Directory where Boltz writes prediction outputs.
        structure_ids: List of structure IDs to collect results for.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.

    Returns:
        BoltzBatchResult with per-structure results for outputs found.
    """
    n_submitted = len(structure_ids)

    if n_submitted == 0:
        logger.info("No structures to process, skipping Boltz subprocess.")
        return BoltzBatchResult(results={}, n_submitted=0, returncode=0, error_msg=None)

    cmd = build_boltz_command(
        yaml_path=input_dir,
        out_dir=out_dir,
        model=model,
        devices=devices,
        accelerator=accelerator,
        diffusion_samples=diffusion_samples,
        sampling_steps=sampling_steps,
        recycling_steps=recycling_steps,
        use_msa_server=use_msa_server,
        override=False,
    )

    logger.info("Running Boltz on directory ({} structures): {}", n_submitted, " ".join(cmd))

    error_msg = None
    returncode = 0

    try:
        env = _clean_env_for_boltz()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        returncode = proc.returncode

        if returncode != 0:
            stderr = proc.stderr
            if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
                error_msg = (
                    f"Boltz OOM: GPU memory exhaustion during batch prediction. "
                    f"Re-run to process remaining structures, or reduce max_length / increase GPU memory.\n"
                    f"stderr: {stderr[-500:]}"
                )
                logger.error(error_msg)
            else:
                error_msg = (
                    f"Boltz failed with return code {returncode}\n"
                    f"stderr: {stderr}\nstdout: {proc.stdout}"
                )
                logger.error(error_msg)

    except Exception as e:
        error_msg = str(e)
        returncode = -1
        logger.error("Boltz subprocess exception: {}", e)

    # Collect results for whatever outputs exist
    results: dict[str, BoltzResult] = {}
    for sid in structure_ids:
        npz_path = find_plddt_npz(out_dir, sid)
        if npz_path is None:
            continue

        plddt = np.load(npz_path)["plddt"]

        conf_json = None
        json_path = find_confidence_json(out_dir, sid)
        if json_path is not None:
            with open(json_path) as f:
                conf_json = json.load(f)

        results[sid] = BoltzResult(
            structure_id=sid,
            plddt=plddt,
            confidence_json=conf_json,
            success=True,
            error_msg=None,
        )

    n_found = len(results)
    if error_msg and n_found > 0:
        error_msg += f"\n{n_found} of {n_submitted} structures completed before failure."

    logger.info(
        "Boltz batch complete: {}/{} structures produced pLDDT (returncode={})",
        n_found, n_submitted, returncode,
    )

    return BoltzBatchResult(
        results=results,
        n_submitted=n_submitted,
        returncode=returncode,
        error_msg=error_msg,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boltz_runner.py::TestRunBoltzPredictDir -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quality_graft/data/boltz_runner.py tests/test_boltz_runner.py
git commit -m "Add run_boltz_predict_dir() for batch Boltz predictions"
```

---

### Task 3: Test OOM detection

**Files:**
- Test: `tests/test_boltz_runner.py` (extend)

- [ ] **Step 1: Write the OOM detection test**

Append to `TestRunBoltzPredictDir` in `tests/test_boltz_runner.py`:

```python
    def test_oom_detection(self, tmp_path):
        """OOM errors produce specific error message."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq_A.yaml").write_text("dummy")
        structure_ids = ["1ubq_A"]

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert result.returncode == 1
        assert "OOM" in result.error_msg
        assert "GPU memory exhaustion" in result.error_msg

    def test_empty_directory(self, tmp_path):
        """No structures submitted skips subprocess entirely."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        with patch("quality_graft.data.boltz_runner.subprocess.run") as mock_run:
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=[],
            )
            mock_run.assert_not_called()

        assert result.n_submitted == 0
        assert len(result.results) == 0
        assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_boltz_runner.py -v`
Expected: All PASS (these test against the implementation from Task 2)

- [ ] **Step 3: Commit**

```bash
git add tests/test_boltz_runner.py
git commit -m "Add OOM detection and edge case tests for batch Boltz runner"
```

---

## Chunk 2: Restructure `_run_boltz_pass()` in datamodule.py and update tests

### Task 4: Restructure `_run_boltz_pass()` into three phases

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:101-197`
- Test: `tests/test_datamodule.py` (will update in Task 5)

- [ ] **Step 1: Rewrite `_run_boltz_pass()` and remove `_run_boltz_for_structure()`**

Replace `_run_boltz_pass()` and `_run_boltz_for_structure()` (lines 101-197) in `src/quality_graft/data/datamodule.py` with:

```python
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

    def _prepare_boltz_yaml(self, structure_id: str, pdb_code: str) -> Optional[Path]:
        """Parse CIF and write Boltz input YAML. Returns yaml_path or None on failure."""
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
        yaml_path = self.boltz_inputs_dir / f"{structure_id}.yaml"
        yaml_path.write_text(yaml_content)
        return yaml_path
```

Also update the import at the top of `datamodule.py` — remove the `run_boltz_predict` import (line 25) since it's no longer used at module level. The new `run_boltz_predict_dir` import is done inline in `_run_boltz_pass()` to keep the lazy-import pattern.

Remove this line:
```python
from quality_graft.data.boltz_runner import run_boltz_predict
```

- [ ] **Step 2: Run existing tests to see them fail (expected — mocks are stale)**

Run: `pytest tests/test_datamodule.py -v`
Expected: FAIL — `_run_boltz_for_structure` no longer exists

- [ ] **Step 3: Commit the restructured datamodule (tests will be fixed in next task)**

```bash
git add src/quality_graft/data/datamodule.py
git commit -m "Restructure _run_boltz_pass() into three-phase directory mode"
```

---

### Task 5: Update datamodule tests

**Files:**
- Modify: `tests/test_datamodule.py`

- [ ] **Step 1: Rewrite both test classes**

Replace the full contents of `tests/test_datamodule.py` with:

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
from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult


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


def _make_batch_result(structure_plddt_map: dict[str, np.ndarray | None]) -> BoltzBatchResult:
    """Build a BoltzBatchResult from a {structure_id: plddt_array} dict."""
    results = {}
    for sid, plddt in structure_plddt_map.items():
        if plddt is not None:
            results[sid] = BoltzResult(
                structure_id=sid,
                plddt=plddt,
                confidence_json=None,
                success=True,
                error_msg=None,
            )
    return BoltzBatchResult(
        results=results,
        n_submitted=len(structure_plddt_map),
        returncode=0,
        error_msg=None,
    )


class TestBoltzPassSkipsExisting:
    """Test that the Boltz pass skips graphs that already have pLDDT."""

    def test_skip_if_plddt_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            graph = _make_fake_graph("test_pdb", has_plddt=True)
            torch.save(graph, processed / "test_pdb.pt")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )

            with patch("quality_graft.data.boltz_runner.run_boltz_predict_dir") as mock_batch:
                dm._run_boltz_pass(["test_pdb.pt"])
                mock_batch.assert_not_called()


class TestBoltzPassProcessesNew:
    """Test that the Boltz pass runs on graphs without pLDDT."""

    def test_processes_new_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            graph = _make_fake_graph("test_pdb", n_residues=5, has_plddt=False)
            torch.save(graph, processed / "test_pdb.pt")

            # Create a dummy CIF file so _prepare_boltz_yaml doesn't fail
            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            fake_result = _make_batch_result({"test_pdb": fake_plddt})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.boltz_runner.run_boltz_predict_dir", return_value=fake_result):
                dm._run_boltz_pass(["test_pdb.pt"])

            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt")
            assert hasattr(updated, "plddt_bin")
            assert updated.plddt.shape[0] == 5
            assert updated.plddt_bin.shape[0] == 5
            assert updated.plddt_bin.dtype == torch.long


class TestBoltzPassPartialFailure:
    """Test that partial Boltz failures are handled gracefully."""

    def test_missing_result_counted_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            # Two structures, only one gets a result
            for sid in ["good_A", "bad_B"]:
                graph = _make_fake_graph(sid, n_residues=5, has_plddt=False)
                torch.save(graph, processed / f"{sid}.pt")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            fake_result = _make_batch_result({"good_A": fake_plddt, "bad_B": None})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.boltz_runner.run_boltz_predict_dir", return_value=fake_result):
                dm._run_boltz_pass(["good_A.pt", "bad_B.pt"])

            good = torch.load(processed / "good_A.pt", weights_only=False)
            bad = torch.load(processed / "bad_B.pt", weights_only=False)

            assert hasattr(good, "plddt")
            assert not hasattr(bad, "plddt") or bad.plddt is None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_datamodule.py -v`
Expected: All PASS

- [ ] **Step 3: Run the full test suite to check nothing else broke**

Run: `pytest tests/ -v`
Expected: All PASS (or only `heavy`-marked tests skipped)

- [ ] **Step 4: Commit**

```bash
git add tests/test_datamodule.py
git commit -m "Update datamodule tests for directory-mode Boltz batch processing"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite one more time**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify the import chain works end-to-end**

Run: `python -c "from quality_graft.data.datamodule import QualityGraftDataModule; from quality_graft.data.boltz_runner import run_boltz_predict_dir, BoltzBatchResult; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Final commit if any remaining changes**

```bash
git status
# If clean, no commit needed
```
