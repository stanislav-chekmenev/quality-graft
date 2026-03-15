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

**`build_boltz_command`** — add two parameters:
- `num_workers: int = 2` → appends `--num_workers <N>` (Boltz dataloader workers)
- `preprocessing_threads: int | None = None` → appends `--preprocessing-threads <N>` if set

**`run_boltz_predict_dir`**:
- Remove `cuda_device` parameter and the `CUDA_VISIBLE_DEVICES` injection logic
- Add `num_workers: int = 2` and `preprocessing_threads: int | None = None` parameters, forwarded to `build_boltz_command`

**`run_boltz_predict`** (single-file version):
- Add `num_workers` and `preprocessing_threads` params for consistency, forwarded to `build_boltz_command`

### 2. `src/quality_graft/data/datamodule.py`

**`_run_boltz_pass`**:
- Remove `ThreadPoolExecutor` / `as_completed` usage
- Remove `num_boltz_workers` config read
- Replace with sequential `for` loop over chunks
- Each chunk calls `run_boltz_predict_dir` with `devices=num_devices` (all GPUs used by a single process)
- Remove round-robin `cuda_device = idx % num_devices` GPU assignment
- Keep per-chunk CSV save and progress logging

**`boltz_kwargs` dict**:
- `"devices"` reads `num_devices` from config (was hardcoded to 1)
- Add `"num_workers"` from config
- Add `"preprocessing_threads"` from config

**Imports**: Remove `concurrent.futures.ThreadPoolExecutor` and `as_completed`.

### 3. Config files

**`configs/data/dataset.yaml`** and **`configs/data/dataset_monomers_len_128.yaml`**:
- Remove `num_boltz_workers`
- Add `num_workers: 2` (Boltz dataloader workers)
- Add `preprocessing_threads: null` (null = Boltz default of `cpu_count()`)
- Keep `chunk_size`, `num_devices`, `timeout_per_structure`

### 4. `scripts/preprocess_full.sbatch`

- Remove `data.boltz.num_boltz_workers=5` and `data.boltz.chunk_size=5` overrides
- Add `data.boltz.num_workers=4` and `data.boltz.preprocessing_threads=8`

## What stays the same

- Chunk-based processing with per-chunk CSV saves (crash resilience)
- YAML preparation logic
- Output parsing (`_find_boltz_output`, `find_plddt_npz`, `find_confidence_json`)
- pLDDT merging into `.pt` files
- `BoltzResult` / `BoltzBatchResult` dataclasses
- `_clean_env_for_boltz()` environment cleaning
- OOM detection and error handling
