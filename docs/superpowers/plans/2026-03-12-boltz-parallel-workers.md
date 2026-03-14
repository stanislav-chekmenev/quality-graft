# Boltz Parallel Workers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the Boltz pLDDT preprocessing pass using `ThreadPoolExecutor` with chunked structures and incremental saves.

**Architecture:** Split structures into chunks of `chunk_size`, submit each chunk to a thread pool that calls the existing `run_boltz_predict_dir()` with per-chunk input/output directories. As each chunk completes via `as_completed`, merge pLDDT into `.pt` files immediately in the main thread. This replaces the single-invocation approach with parallel subprocess execution.

**Tech Stack:** Python stdlib `concurrent.futures.ThreadPoolExecutor`, `shutil`, existing `run_boltz_predict_dir()` from `boltz_runner.py`.

**Spec:** `docs/superpowers/specs/2026-03-12-boltz-parallel-workers-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/quality_graft/data/datamodule.py` | Modify | Rewrite `_run_boltz_pass()` for parallel chunked execution, add `output_dir` param to `_prepare_boltz_yaml` |
| `configs/data/dataset_monomers_len_128_frac_010.yaml` | Modify | Add `num_boltz_workers` and `chunk_size` to boltz block |
| `tests/test_datamodule.py` | Modify | Update existing tests for chunked execution, add new parallel-specific tests |

---

## Chunk 1: Config and `_prepare_boltz_yaml` update

### Task 1: Add config parameters

**Files:**
- Modify: `configs/data/dataset_monomers_len_128_frac_010.yaml:26-33`

- [ ] **Step 1: Add num_boltz_workers and chunk_size to boltz config**

```yaml
boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 50
  recycling_steps: 0
  devices: 1
  accelerator: gpu
  use_msa_server: false
  num_boltz_workers: 2
  chunk_size: 10
```

- [ ] **Step 2: Commit**

```bash
git add configs/data/dataset_monomers_len_128_frac_010.yaml
git commit -m "feat: add num_boltz_workers and chunk_size to boltz config"
```

### Task 2: Update `_prepare_boltz_yaml` to accept output_dir

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:195-216`
- Test: `tests/test_datamodule.py`

- [ ] **Step 1: Write failing test for output_dir parameter**

Add to `tests/test_datamodule.py`:

```python
class TestPrepareYamlOutputDir:
    """Test that _prepare_boltz_yaml writes to custom output_dir."""

    def test_writes_to_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)
            custom_dir = Path(tmpdir) / "boltz_work" / "inputs" / "chunk_000"
            custom_dir.mkdir(parents=True)

            # Write a minimal valid CIF
            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"use_msa_server": False},
            )

            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[{"sequence": "ACGT"}]), \
                 patch("quality_graft.data.datamodule.chains_to_boltz_yaml", return_value="dummy_yaml"):
                result = dm._prepare_boltz_yaml("test_A", "test", output_dir=custom_dir)

            assert result is not None
            assert result.parent == custom_dir
            assert (custom_dir / "test_A.yaml").exists()

    def test_defaults_to_boltz_inputs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"use_msa_server": False},
            )

            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[{"sequence": "ACGT"}]), \
                 patch("quality_graft.data.datamodule.chains_to_boltz_yaml", return_value="dummy_yaml"):
                result = dm._prepare_boltz_yaml("test_A", "test")

            assert result is not None
            assert result.parent == boltz_inputs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datamodule.py::TestPrepareYamlOutputDir -v`
Expected: FAIL — `_prepare_boltz_yaml() got an unexpected keyword argument 'output_dir'`

- [ ] **Step 3: Update `_prepare_boltz_yaml` to accept output_dir**

In `src/quality_graft/data/datamodule.py`, replace the method signature and the yaml_path line:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datamodule.py::TestPrepareYamlOutputDir -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/test_datamodule.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule.py
git commit -m "feat: add output_dir param to _prepare_boltz_yaml"
```

---

## Chunk 2: Rewrite `_run_boltz_pass()` for parallel chunked execution

### Task 3: Write tests for chunked parallel execution

**Files:**
- Modify: `tests/test_datamodule.py`

These tests validate the new `_run_boltz_pass()` behavior. They mock `run_boltz_predict_dir` at the module level (which works regardless of threading).

- [ ] **Step 1: Add chunking test**

Add to `tests/test_datamodule.py`:

```python
class TestParallelBoltzPass:
    """Test parallel chunked Boltz execution."""

    def _setup_structures(self, tmpdir, structure_ids, n_residues=5):
        """Helper to create .pt files and raw CIFs for testing."""
        processed = Path(tmpdir) / "processed"
        processed.mkdir(exist_ok=True)
        raw = Path(tmpdir) / "raw"
        raw.mkdir(exist_ok=True)
        boltz_work = Path(tmpdir) / "boltz_work"
        boltz_work.mkdir(exist_ok=True)
        (boltz_work / "inputs").mkdir(exist_ok=True)

        for sid in structure_ids:
            graph = _make_fake_graph(sid, n_residues=n_residues, has_plddt=False)
            torch.save(graph, processed / f"{sid}.pt")
            pdb_code = sid.split("_")[0]
            (raw / f"{pdb_code}.cif").write_text("dummy")

        return processed

    def test_structures_split_into_chunks(self):
        """Verify run_boltz_predict_dir is called once per chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(25)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "num_boltz_workers": 2,
                    "chunk_size": 10,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir) as mock_batch:
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # 25 structures / chunk_size 10 = 3 chunks
            assert mock_batch.call_count == 3

            # Verify all structures got pLDDT
            processed = Path(tmpdir) / "processed"
            for sid in sids:
                graph = torch.load(processed / f"{sid}.pt", weights_only=False)
                assert hasattr(graph, "plddt"), f"{sid} missing plddt"
                assert hasattr(graph, "plddt_bin"), f"{sid} missing plddt_bin"

    def test_chunk_failure_doesnt_block_others(self):
        """One chunk failing should not prevent other chunks from saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(20)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "num_boltz_workers": 2,
                    "chunk_size": 10,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                # Fail chunk_000 deterministically by directory name
                if input_dir.name == "chunk_000":
                    return BoltzBatchResult(
                        results={}, n_submitted=len(structure_ids),
                        returncode=1, error_msg="Boltz OOM: GPU memory exhaustion",
                    )
                # All other chunks succeed
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # At least some structures should have pLDDT (from the successful chunk)
            processed = Path(tmpdir) / "processed"
            labeled = 0
            for sid in sids:
                graph = torch.load(processed / f"{sid}.pt", weights_only=False)
                if hasattr(graph, "plddt") and graph.plddt is not None:
                    labeled += 1
            assert labeled == 10  # One chunk of 10 succeeded

    def test_each_chunk_gets_own_directories(self):
        """Verify each chunk gets unique input and output dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(5)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "num_boltz_workers": 2,
                    "chunk_size": 3,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            seen_input_dirs = []
            seen_out_dirs = []

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                seen_input_dirs.append(input_dir)
                seen_out_dirs.append(out_dir)
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # 5 structures / chunk_size 3 = 2 chunks
            assert len(seen_input_dirs) == 2
            assert len(set(seen_input_dirs)) == 2  # All unique
            assert len(set(seen_out_dirs)) == 2  # All unique
            # Verify chunk naming
            for d in seen_input_dirs:
                assert d.name.startswith("chunk_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datamodule.py::TestParallelBoltzPass -v`
Expected: FAIL — tests call the old single-invocation `_run_boltz_pass` which doesn't use chunks

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_datamodule.py
git commit -m "test: add parallel Boltz pass tests (red)"
```

### Task 4: Rewrite `_run_boltz_pass()` for parallel chunked execution

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:1-17` (imports)
- Modify: `src/quality_graft/data/datamodule.py:100-193` (`_run_boltz_pass`)

- [ ] **Step 1: Add imports at the top of datamodule.py**

Add `shutil` and `concurrent.futures` to the imports section. Also move the `run_boltz_predict_dir` import to top-level (currently a lazy import inside `_run_boltz_pass`):

```python
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
```

- [ ] **Step 2: Replace `_run_boltz_pass` method**

Replace the entire `_run_boltz_pass` method (lines 100-193) with:

```python
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
        boltz_kwargs = {
            "model": self.boltz_config.get("model", "boltz1"),
            "devices": self.boltz_config.get("devices", 1),
            "accelerator": self.boltz_config.get("accelerator", "gpu"),
            "diffusion_samples": self.boltz_config.get("diffusion_samples", 1),
            "sampling_steps": self.boltz_config.get("sampling_steps", 200),
            "recycling_steps": self.boltz_config.get("recycling_steps", 3),
            "use_msa_server": self.boltz_config.get("use_msa_server", False),
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
```

- [ ] **Step 3: Run new parallel tests**

Run: `pytest tests/test_datamodule.py::TestParallelBoltzPass -v`
Expected: PASS

- [ ] **Step 4: Update existing tests for the new chunked behavior**

The existing tests in `TestBoltzPassProcessesNew` and `TestBoltzPassPartialFailure` mock `run_boltz_predict_dir` at the old import path (`quality_graft.data.boltz_runner.run_boltz_predict_dir`). Since the import is now at module level, the mock target changes to `quality_graft.data.datamodule.run_boltz_predict_dir`. Additionally, the mocked function is now called per-chunk, so `side_effect` (callable) is needed instead of `return_value`.

Update `TestBoltzPassProcessesNew.test_processes_new_graph`:

```python
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

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass(["test_pdb.pt"])

            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt")
            assert hasattr(updated, "plddt_bin")
            assert updated.plddt.shape[0] == 5
            assert updated.plddt_bin.shape[0] == 5
            assert updated.plddt_bin.dtype == torch.long
```

Update `TestBoltzPassPartialFailure.test_missing_result_counted_as_failed`:

```python
    def test_missing_result_counted_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            for sid in ["good_A", "bad_B"]:
                graph = _make_fake_graph(sid, n_residues=5, has_plddt=False)
                torch.save(graph, processed / f"{sid}.pt")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                result_map = {}
                for sid in structure_ids:
                    if sid == "good_A":
                        result_map[sid] = fake_plddt.copy()
                    else:
                        result_map[sid] = None
                return _make_batch_result(result_map)

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass(["good_A.pt", "bad_B.pt"])

            good = torch.load(processed / "good_A.pt", weights_only=False)
            bad = torch.load(processed / "bad_B.pt", weights_only=False)

            assert hasattr(good, "plddt")
            assert not hasattr(bad, "plddt") or bad.plddt is None
```

Update `TestBoltzPassSkipsExisting.test_skip_if_plddt_present` — the mock target also changes:

```python
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

            with patch("quality_graft.data.datamodule.run_boltz_predict_dir") as mock_batch:
                dm._run_boltz_pass(["test_pdb.pt"])
                mock_batch.assert_not_called()
```

- [ ] **Step 5: Run all datamodule tests**

Run: `pytest tests/test_datamodule.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule.py
git commit -m "feat: parallel chunked Boltz pass with ThreadPoolExecutor and incremental saves"
```

---

## Chunk 3: Integration verification

### Task 5: Run full test suite and verify

- [ ] **Step 1: Run all unit tests**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Verify debug_preprocess.sh works with override**

The debug script should work as-is. To test with explicit parallelism params, run:

```bash
bash scripts/debug_preprocess.sh
```

Or with overrides:

```bash
python scripts/train.py mode=preprocess data.data_dir=data/debug data.local_only=true data.boltz.num_boltz_workers=1 data.boltz.chunk_size=2 data.boltz.sampling_steps=20 data.boltz.recycling_steps=1 training.batch_size=1 training.num_workers=1 training.max_length=128
```

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -u
git commit -m "fix: address test/integration issues from parallel Boltz pass"
```
