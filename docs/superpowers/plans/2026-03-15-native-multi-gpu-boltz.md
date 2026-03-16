# Native Multi-GPU Boltz Preprocessing — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ThreadPoolExecutor-based multi-subprocess Boltz preprocessing with Boltz's native `--devices N` multi-GPU support for proper GPU utilization.

**Architecture:** Remove parallel subprocess spawning. Each chunk now runs as a single `boltz predict` call with `--devices N`, processing sequentially across chunks for crash resilience. New CLI params (`--num_workers`, `--preprocessing-threads`, `--max_parallel_samples`) are exposed through Hydra config.

**Tech Stack:** Python, Hydra, subprocess, Boltz CLI

**Spec:** `docs/superpowers/specs/2026-03-15-native-multi-gpu-boltz-design.md`

---

## Chunk 1: boltz_runner.py — CLI command builder and runner updates

### Task 1: Update `build_boltz_command` signature and body

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py:70-125`

- [ ] **Step 1: Add three new parameters to `build_boltz_command`**

Add `num_workers`, `preprocessing_threads`, and `max_parallel_samples` to the function signature and append the corresponding CLI flags:

```python
def build_boltz_command(
    yaml_path: Path,
    out_dir: Path,
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    override: bool = False,
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
) -> list[str]:
    """Build the boltz predict CLI command as a list of strings.

    Args:
        yaml_path: Path to the input YAML file for Boltz.
        out_dir: Directory where Boltz writes prediction outputs.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        override: Whether to override existing results.
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        Command as a list of strings suitable for subprocess.run().
    """
    cmd = [
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--model",
        model,
        "--devices",
        str(devices),
        "--accelerator",
        accelerator,
        "--diffusion_samples",
        str(diffusion_samples),
        "--sampling_steps",
        str(sampling_steps),
        "--recycling_steps",
        str(recycling_steps),
        "--num_workers",
        str(num_workers),
    ]

    if preprocessing_threads is not None:
        cmd.extend(["--preprocessing-threads", str(preprocessing_threads)])

    if max_parallel_samples is not None:
        cmd.extend(["--max_parallel_samples", str(max_parallel_samples)])

    if use_msa_server:
        cmd.append("--use_msa_server")

    if override:
        cmd.append("--override")

    return cmd
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `pytest tests/test_datamodule.py -v`
Expected: All tests PASS (build_boltz_command changes are backward-compatible via defaults)

- [ ] **Step 3: Commit**

```bash
git add src/quality_graft/data/boltz_runner.py
git commit -m "Add num_workers, preprocessing_threads, max_parallel_samples to build_boltz_command"
```

### Task 2: Update `run_boltz_predict` (single-file runner)

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py:186-285`

- [ ] **Step 1: Add new params, forward to `build_boltz_command`**

```python
def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    override: bool = False,
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
) -> BoltzResult:
    """Run boltz predict as a subprocess and parse results.

    Args:
        yaml_path: Path to the input YAML file. The stem is used as structure_id.
        out_dir: Directory where Boltz writes prediction outputs.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        override: Whether to override existing results.
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        BoltzResult with pLDDT array on success, or error information on failure.
    """
    structure_id = yaml_path.stem
    cmd = build_boltz_command(
        yaml_path,
        out_dir,
        model,
        devices,
        accelerator,
        diffusion_samples,
        sampling_steps,
        recycling_steps,
        use_msa_server,
        override,
        num_workers,
        preprocessing_threads,
        max_parallel_samples,
    )
```

The rest of the function body stays unchanged.

- [ ] **Step 2: Commit**

```bash
git add src/quality_graft/data/boltz_runner.py
git commit -m "Add new params to run_boltz_predict"
```

### Task 3: Update `run_boltz_predict_dir` — remove cuda_device, add new params

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py:288-444`

- [ ] **Step 1: Remove `cuda_device`, add new params, update error message**

New signature and key changes:

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
    timeout: int | None = None,
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
) -> BoltzBatchResult:
    """Run boltz predict on a directory of YAMLs and collect results.

    Passes the entire input_dir to a single `boltz predict` invocation
    with native multi-GPU support via --devices. After the subprocess
    finishes (or crashes), iterates over structure_ids and collects
    whatever pLDDT outputs exist.

    Args:
        input_dir: Directory containing YAML files for Boltz.
        out_dir: Directory where Boltz writes prediction outputs.
        structure_ids: List of structure IDs to collect results for.
        model: Model name (default: "boltz1").
        devices: Number of devices to use (passed as --devices to Boltz).
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        timeout: Max seconds to wait for the subprocess. None means no limit.
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        BoltzBatchResult with per-structure results for outputs found.
    """
```

In the body, update the `build_boltz_command` call to forward new params:

```python
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
        num_workers=num_workers,
        preprocessing_threads=preprocessing_threads,
        max_parallel_samples=max_parallel_samples,
    )
```

Remove the `CUDA_VISIBLE_DEVICES` injection (lines 352-353):

```python
    # REMOVE these two lines:
    #     if cuda_device is not None:
    #         env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
```

Update timeout error message (lines 381-384):

```python
    except subprocess.TimeoutExpired as e:
        error_msg = (
            f"Boltz subprocess timed out after {timeout}s. "
            f"Reduce chunk_size or increase timeout."
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/quality_graft/data/boltz_runner.py
git commit -m "Remove cuda_device, add native multi-GPU params to run_boltz_predict_dir"
```

## Chunk 2: datamodule.py + tests — sequential chunk processing

### Task 4: Replace ThreadPoolExecutor with sequential loop and update tests

**Files:**
- Modify: `src/quality_graft/data/datamodule.py:17` (remove import)
- Modify: `src/quality_graft/data/datamodule.py:250-449` (rewrite `_run_boltz_pass`)
- Modify: `tests/test_datamodule.py:214-341` (update test class)

- [ ] **Step 1: Remove ThreadPoolExecutor import**

Replace line 17:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```
with nothing (delete the line).

- [ ] **Step 2: Rewrite `_run_boltz_pass`**

Replace the entire method (lines 250-449) with:

```python
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
            if structure_id in plddt_set:
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
```

- [ ] **Step 3: Update `TestParallelBoltzPass` in `tests/test_datamodule.py`**

Rename the class and remove `"num_boltz_workers"` from all `boltz_config` dicts. The mock's `**kwargs` will no longer include `cuda_device` — no explicit assertion changes needed since the mocks use `**kwargs`.

Rename class (line 214):
```python
class TestChunkedBoltzPass:
    """Test sequential chunked Boltz execution."""
```

In `test_structures_split_into_chunks` (line 243), change boltz_config:
```python
            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 10,
                },
            )
```

In `test_chunk_failure_doesnt_block_others` (line 276), change boltz_config:
```python
            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 10,
                },
            )
```

In `test_each_chunk_gets_own_directories` (line 315), change boltz_config:
```python
            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 3,
                },
            )
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_datamodule.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/quality_graft/data/datamodule.py tests/test_datamodule.py
git commit -m "Replace ThreadPoolExecutor with sequential chunked Boltz processing"
```

## Chunk 3: Configs and scripts

### Task 5: Update config files

**Files:**
- Modify: `configs/data/dataset.yaml:22-31`
- Modify: `configs/data/dataset_monomers_len_128.yaml:26-37`

- [ ] **Step 1: Update `configs/data/dataset.yaml`**

Replace the boltz section (lines 22-31) with:
```yaml
boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 200
  recycling_steps: 3
  accelerator: gpu
  use_msa_server: false
  timeout_per_structure: 300  # seconds; chunk timeout = chunk_size * this + 120s
  num_devices: 1  # Number of GPUs for Boltz --devices flag
  num_workers: 2  # Boltz dataloader workers
  preprocessing_threads: null  # null = Boltz default (cpu_count)
  max_parallel_samples: null  # null = Boltz default (5)
```

- [ ] **Step 2: Update `configs/data/dataset_monomers_len_128.yaml`**

Replace the boltz section (lines 26-37) with:
```yaml
boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 50
  recycling_steps: 1
  accelerator: gpu
  use_msa_server: false
  chunk_size: 10
  timeout_per_structure: 300  # seconds; chunk timeout = chunk_size * this + 120s
  num_devices: 1  # Number of GPUs for Boltz --devices flag
  num_workers: 2  # Boltz dataloader workers
  preprocessing_threads: null  # null = Boltz default (cpu_count)
  max_parallel_samples: null  # null = Boltz default (5)
```

- [ ] **Step 3: Commit**

```bash
git add configs/data/dataset.yaml configs/data/dataset_monomers_len_128.yaml
git commit -m "Remove devices/num_boltz_workers, add num_workers/preprocessing_threads/max_parallel_samples"
```

### Task 6: Update scripts

**Files:**
- Modify: `scripts/preprocess_full.sbatch:54-63`
- Modify: `scripts/debug_preprocess.sh:21-33`

- [ ] **Step 1: Update `scripts/preprocess_full.sbatch`**

Replace lines 54-63 with:
```bash
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.local_only=false \
    data.data_dir="$SCRATCH_DIR" \
    data.selector_num_workers=16 \
    data.boltz.num_devices=$NUM_DEVICES \
    data.boltz.num_workers=4 \
    data.boltz.preprocessing_threads=8 \
    data.boltz.timeout_per_structure=180 \
    training.max_length=128
```

- [ ] **Step 2: Update `scripts/debug_preprocess.sh`**

Replace lines 21-33 with:
```bash
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$DEBUG_DIR" \
    data.local_only=true \
    data.selector_num_workers=4 \
    data.boltz.chunk_size=1 \
    data.num_workers=4 \
    data.boltz.sampling_steps=20 \
    data.boltz.recycling_steps=1 \
    training.batch_size=1 \
    training.num_workers=1 \
    training.max_length=128
```

- [ ] **Step 3: Run tests one final time**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/preprocess_full.sbatch scripts/debug_preprocess.sh
git commit -m "Update preprocessing scripts for native multi-GPU Boltz"
```
