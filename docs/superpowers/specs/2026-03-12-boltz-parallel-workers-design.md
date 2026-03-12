# Boltz Parallel Workers Design

**Date:** 2026-03-12
**Status:** Approved

## Problem

The current Boltz preprocessing pipeline runs a single `boltz predict` subprocess that processes structures sequentially. This underutilizes the GPU and makes large runs (500-1000+ structures) very slow. A previous 1000-protein run failed due to node disconnect with no data saved at all — the all-or-nothing save model means any interruption loses all progress.

## Solution

Parallelize the Boltz pass using `concurrent.futures.ThreadPoolExecutor`. Structures are split into small chunks (~10 each), with multiple chunks processed simultaneously by separate `boltz predict` subprocesses sharing the same GPU. Results are saved incrementally as each chunk completes.

`ThreadPoolExecutor` is used rather than `ProcessPoolExecutor` because each worker simply calls `subprocess.run()` and waits — there is no CPU-bound work, so threads avoid the process spawn overhead and pickling constraints.

## Design

### Chunking and Worker Architecture

After Phase 1 (YAML preparation), `submitted_ids` are split into chunks of `chunk_size` (default 10). Each chunk gets its own input subdirectory and output subdirectory:

- Input: `boltz_inputs_dir/chunk_000/`, `boltz_inputs_dir/chunk_001/`, ...
- Output: `boltz_work_dir/chunk_000/`, `boltz_work_dir/chunk_001/`, ...

During Phase 1, structures are assigned to chunks round-robin, and `_prepare_boltz_yaml` is modified to accept an optional `output_dir` parameter (defaults to `self.boltz_inputs_dir` for backwards compatibility). YAMLs are written directly into chunk input directories.

The stale cleanup step at the start of Phase 1 removes old `chunk_*` subdirectories under both `boltz_inputs_dir/` and `boltz_work_dir/` using `shutil.rmtree()`, replacing the current `*.yaml` glob cleanup.

All chunks are submitted to a `ThreadPoolExecutor(max_workers=num_boltz_workers)`. Each worker calls the existing `run_boltz_predict_dir()` on its chunk's input directory with its chunk's output directory — no changes to `boltz_runner.py` needed.

**Per-chunk data flow**: A dict maps each `Future` to a tuple of `(chunk_idx: int, chunk_structure_ids: list[str])`. This is built during submission and used in the `as_completed` loop to know which structures to merge for each completed chunk. Critically, each chunk's `structure_ids` list is passed to `run_boltz_predict_dir()` so it only looks for outputs belonging to that chunk — passing the full `submitted_ids` would cause spurious "not found" warnings.

**Phase 3 runs in the main thread** via the `as_completed` loop. Each chunk's structure sets are disjoint (assigned during chunking), so there are no concurrent `.pt` file writes. The main thread merges pLDDT into `.pt` files for one chunk at a time.

**Result aggregation**: Three global counters (`n_labeled`, `n_failed`, `n_skipped`) are maintained in the main thread and updated after each chunk completes. Per-chunk results from `BoltzBatchResult` are processed inline — no aggregation into a combined dict.

### Boltz Output Directory Layout

Each chunk's Boltz output lands under its dedicated output directory. `run_boltz_predict_dir()` already handles the Boltz output layout: it checks `out_dir/boltz_results_{input_dir.name}/predictions/{sid}/` first, then falls back to `out_dir/predictions/{sid}/`. With chunk input dirs named `chunk_000`, the primary lookup path becomes `boltz_work/chunk_000/boltz_results_chunk_000/predictions/{sid}/`.

### GPU Memory Considerations

Each `boltz predict` subprocess loads its own copy of the Boltz model (~3GB) into GPU memory. With `num_boltz_workers=N`, peak GPU memory is approximately `N * (model_size + activation_size)`. For small proteins (max_length=128), activation memory is small, so model weight duplication dominates.

- **Default 2 workers**: safe for most GPUs (2 * ~3GB = ~6GB model memory). The user should increase based on available GPU memory.
- **Tuning guideline**: `(GPU_memory_GB - 4) / 3` gives a rough upper bound for workers. E.g., A100 80GB → ~25 workers max, V100 32GB → ~9 workers max.
- **OOM is safe**: if too many workers cause OOM, individual chunks fail gracefully (partial results collected, structures retried on re-run via `plddt_bin` skip check).

No automatic GPU memory management is added — the user tunes `num_boltz_workers` to their hardware.

### Config

Two new parameters in `data.boltz`:

- `num_boltz_workers: 2` — number of parallel `boltz predict` subprocesses (increase based on GPU memory)
- `chunk_size: 10` — maximum structures per subprocess

These are added to `configs/data/dataset_monomers_len_128_frac_010.yaml` and flow through the existing `boltz_config` dict to `QualityGraftDataModule`. Accessed in `_run_boltz_pass()` via `self.boltz_config.get("num_boltz_workers", 2)` and `self.boltz_config.get("chunk_size", 10)`. Override via Hydra CLI: `data.boltz.num_boltz_workers=5 data.boltz.chunk_size=10`.

### Error Handling and Resilience

**Per-chunk isolation:** Each chunk runs independently. If one chunk's `boltz predict` crashes (OOM or otherwise), other chunks are unaffected. The `as_completed` loop handles each future individually — a failed chunk logs its error, collects any partial results, and continues.

**Node disconnect resilience:** Saves happen after each chunk completes (~10 structures). A node disconnect loses at most one chunk's worth of work per active worker. On re-run, the existing `plddt_bin` check in Phase 1 skips all previously-saved structures — this is the "retry on re-run" mechanism referenced in OOM log messages.

**No retry logic:** Failed chunks are logged and skipped. Re-running the script picks up where it left off via the skip check. Keeps the code simple.

### Progress Logging

Per-chunk (after each `as_completed` future resolves). Note: chunks complete out of order; the first number is "chunks completed so far", not the chunk index:
```
Chunks done: 3/50 | total labeled: 27/500 (5.4%) | this chunk: 8/10 succeeded, 2 failed
```

OOM chunks:
```
Chunk OOM: 3/10 structures completed before GPU memory exhaustion. Will retry on re-run.
```

Summary at end:
```
Boltz parallel pass complete: 480/500 labeled, 15 failed, 5 skipped (already had pLDDT) | 50 chunks, 2 workers
```

No progress bars — structured log lines that work well in SLURM output files.

### Files Modified

1. **`src/quality_graft/data/datamodule.py`** — Rewrite `_run_boltz_pass()`:
   - Phase 1: prepare YAMLs directly into chunk subdirectories, clean stale chunk dirs first
   - Phase 2: `ThreadPoolExecutor` submits `run_boltz_predict_dir()` per chunk
   - Phase 3: `as_completed` loop in main thread — merge pLDDT into `.pt` files per chunk, update counters, log progress

2. **`configs/data/dataset_monomers_len_128_frac_010.yaml`** — Add `num_boltz_workers: 2` and `chunk_size: 10` to boltz block

### Files NOT Modified

- **`boltz_runner.py`** — `run_boltz_predict_dir()` already does what each worker needs
- **`train.py`** — `boltz_config` dict passes through, new keys flow automatically
- **Shell scripts** — no changes needed, override via Hydra CLI if desired

### Testing

Existing tests in `tests/test_datamodule.py` mock `run_boltz_predict_dir`. After this change, the mock needs to handle:
- Multiple calls (one per chunk) rather than a single call
- The `ThreadPoolExecutor` context — mock at the `run_boltz_predict_dir` function level, which works regardless of whether it's called from threads
- Per-chunk input/output directories

New test cases:
- Chunking logic: N structures split into correct number of chunks
- Parallel dispatch: verify `run_boltz_predict_dir` called once per chunk with correct arguments
- Incremental save: verify `.pt` files updated after each chunk, not batched
- Partial failure: one chunk fails, others succeed and save correctly

### New Files

None.
