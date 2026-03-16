# Native Multi-GPU Boltz Preprocessing

**Date:** 2026-03-15
**Status:** Draft

## Problem

The current preprocessing pipeline spawns multiple `boltz predict` subprocesses via `ThreadPoolExecutor`, each pinned to a GPU with `CUDA_VISIBLE_DEVICES`. Each subprocess loads the full Boltz model independently, resulting in:

- ~10% GPU utilization (processes block each other on model loading)
- Only ~2 concurrent processes despite requesting 5 workers
- Wasted time on redundant model loads per chunk

## Solution

Use Boltz's native `--devices N` flag to let a single process distribute inference across all GPUs via PyTorch Lightning. Remove the `ThreadPoolExecutor` parallelism. Keep chunked sequential processing for crash resilience (CSV status saved between chunks).

## Changes

### 1. `src/quality_graft/data/boltz_runner.py`

**`build_boltz_command`** — add three parameters:
- `num_workers: int = 2` → appends `--num_workers <N>` (Boltz dataloader workers)
- `preprocessing_threads: int | None = None` → appends `--preprocessing-threads <N>` if set (note: CLI flag uses hyphen, Python param uses underscore)
- `max_parallel_samples: int | None = None` → appends `--max_parallel_samples <N>` if set (controls how many diffusion samples are processed in parallel; helps manage GPU memory)

**`run_boltz_predict_dir`**:
- Remove `cuda_device` parameter and the `CUDA_VISIBLE_DEVICES` injection logic
- Add `num_workers: int = 2`, `preprocessing_threads: int | None = None`, and `max_parallel_samples: int | None = None` parameters, forwarded to `build_boltz_command`
- Update timeout error message: replace "Reduce num_boltz_workers" with "Reduce chunk_size or increase timeout"

**`run_boltz_predict`** (single-file version):
- Add `num_workers`, `preprocessing_threads`, and `max_parallel_samples` params for consistency, forwarded to `build_boltz_command`

### 2. `src/quality_graft/data/datamodule.py`

**`_run_boltz_pass`**:
- Remove `ThreadPoolExecutor` / `as_completed` usage
- Remove `num_boltz_workers` config read
- Replace with sequential `for` loop over chunks
- Each chunk calls `run_boltz_predict_dir` with `devices=num_devices` (all GPUs used by a single process)
- Remove round-robin `cuda_device = idx % num_devices` GPU assignment
- Keep per-chunk CSV save and progress logging
- Update docstring and log messages to reflect sequential processing (remove "parallel" / "workers" references)

**`boltz_kwargs` dict**:
- Remove `"devices"` key (was reading the old `devices: 1` config key)
- Add `"devices"` reading from `num_devices` config key (the GPU count for Boltz `--devices` flag)
- Add `"num_workers"` from config
- Add `"preprocessing_threads"` from config
- Add `"max_parallel_samples"` from config

**Config key consolidation**: The current config has both `devices: 1` and `num_devices: 1`. After this change, only `num_devices` is used (it becomes the `--devices` value). Remove the `devices` key from all config files to avoid confusion.

**Imports**: Remove `concurrent.futures.ThreadPoolExecutor` and `as_completed`.

### 3. Config files

**`configs/data/dataset.yaml`**:
- Remove `devices` key (consolidated into `num_devices`)
- Add `num_workers: 2` (Boltz dataloader workers)
- Add `preprocessing_threads: null` (null = Boltz default of `cpu_count()`)
- Add `max_parallel_samples: null` (null = Boltz default of 5)
- Note: this file has no `num_boltz_workers` or `chunk_size` — no removal needed

**`configs/data/dataset_monomers_len_128.yaml`**:
- Remove `num_boltz_workers` key
- Remove `devices` key (consolidated into `num_devices`)
- Add `num_workers: 2` (Boltz dataloader workers)
- Add `preprocessing_threads: null`
- Add `max_parallel_samples: null`
- Keep `chunk_size`, `num_devices`, `timeout_per_structure`

### 4. `scripts/preprocess_full.sbatch`

- Remove `data.boltz.num_boltz_workers=5` and `data.boltz.chunk_size=5` overrides
- Add `data.boltz.num_workers=4` and `data.boltz.preprocessing_threads=8`

### 5. `scripts/debug_preprocess.sh`

- Remove `data.boltz.num_boltz_workers=2` override (now unused)
- Keep `data.boltz.chunk_size=1` (still valid for debug)

### 6. `tests/test_datamodule.py`

- Remove `"num_boltz_workers"` from all `boltz_config` test dicts
- Remove any `cuda_device` assertions from mock call checks
- Keep `"chunk_size"` in test configs (still used)
- Verify test assertions still hold with sequential loop (mock call order may change from arbitrary `as_completed` order to deterministic sequential order)

## What stays the same

- Chunk-based processing with per-chunk CSV saves (crash resilience)
- `chunk_size` config key and chunked loop structure
- YAML preparation logic
- Output parsing (`_find_boltz_output`, `find_plddt_npz`, `find_confidence_json`)
- pLDDT merging into `.pt` files
- `BoltzResult` / `BoltzBatchResult` dataclasses
- `_clean_env_for_boltz()` environment cleaning
- OOM detection and error handling
- Chunk timeout formula: `chunk_size * timeout_per_structure + 120`
