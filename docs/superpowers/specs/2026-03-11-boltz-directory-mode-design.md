# Boltz Directory Mode — Batch pLDDT Preprocessing

## Problem

The Boltz pLDDT pass in `_run_boltz_pass()` spawns a fresh `boltz predict` subprocess per structure. Each invocation loads the full Boltz model from checkpoint, runs inference for one structure, then exits. For ~1000 structures at 50 diffusion steps, the per-structure model loading dominates wall time.

## Solution

Pass a directory of YAML files to a single `boltz predict` invocation. Boltz loads the model once and processes all structures internally, eliminating per-structure checkpoint loading overhead.

## Design

### Restructured `_run_boltz_pass()` — Three Phases

The current single loop (prepare → run → collect per structure) becomes three distinct phases:

**Phase 1 — Prepare all YAMLs**: Clear `boltz_inputs_dir` to remove stale YAMLs from previous runs. Loop through `.pt` files, skip those with `plddt_bin` already set, parse CIFs, write YAML files to `boltz_inputs_dir`. Build a list of submitted `structure_id`s.

**Phase 2 — Single Boltz invocation**: Call `boltz predict <boltz_inputs_dir>` once with `override=False`. Boltz loads the model once and processes every YAML in the directory. With `override=False`, Boltz skips structures whose outputs already exist from a previous partial run (crash recovery). If no structures need processing, skip entirely.

**Phase 3 — Collect results**: Loop through the submitted `structure_id` list from Phase 1. For each, call `find_plddt_npz(out_dir, structure_id)` to locate the output. If found, validate residue count against the `.pt` graph, store `plddt` + `plddt_bin`, save. If not found, count as failed. Log per-structure results and final summary.

**Error handling**: Even if Boltz returns a non-zero exit code (crash mid-run), Phase 3 still iterates over submitted structure_ids and collects whatever outputs exist. The existing skip logic makes re-runs safe — structures that got labels in Phase 3 are skipped on next run.

### Changes to `boltz_runner.py`

Add one new function `run_boltz_predict_dir()` alongside the existing `run_boltz_predict()`:

- Takes `input_dir` (directory of YAMLs) instead of `yaml_path` (single YAML)
- Same parameters otherwise (model, devices, sampling_steps, etc.)
- Returns a new dataclass `BoltzBatchResult` containing:
  - `results: dict[str, BoltzResult]` — keyed by structure_id, one entry per output found
  - `n_submitted: int` — how many YAMLs were in the directory
  - `returncode: int` — Boltz process exit code
  - `error_msg: str | None` — stderr if non-zero exit
- Accepts a `structure_ids: list[str]` parameter — the list of submitted structure IDs from Phase 1
- Reuses `build_boltz_command()` internally, passing a directory path instead of a file path (Boltz CLI accepts both). Note: `build_boltz_command()` parameter is named `yaml_path` but its body just does `str(yaml_path)`, so it works with directory paths without modification.
- Passes `override=False` so Boltz skips structures with existing outputs (crash recovery)
- After the subprocess finishes (or crashes), iterates over `structure_ids` and calls `find_plddt_npz()` per structure to build a `BoltzResult` for each found output. This reuses the existing lookup logic rather than globbing.

The existing `run_boltz_predict()` and `build_boltz_command()` stay untouched — still useful for single-structure debug runs.

### GPU Memory Handling

Boltz internally manages its own batching when processing a directory. If it hits OOM, the subprocess crashes.

**Detection**: After non-zero exit, check stderr for CUDA OOM indicators (`"CUDA out of memory"`, `"OutOfMemoryError"`).

**Logging**: Log a clear, actionable error: `"Boltz OOM: N of M structures completed before GPU memory exhaustion. Re-run to process remaining structures, or reduce max_length / increase GPU memory."`.

**Graceful degradation**: Collect whatever outputs succeeded before the crash, merge them into `.pt` files, report partial progress. On re-run, completed structures are skipped automatically.

**No retry logic**: Boltz controls its own batching internally, and structures are capped at `max_length: 128`. If OOM happens, it's a signal the user needs to address externally.

## Files Modified

| File | Change |
|---|---|
| `src/quality_graft/data/boltz_runner.py` | Add `BoltzBatchResult` dataclass and `run_boltz_predict_dir()` function with OOM detection |
| `src/quality_graft/data/datamodule.py` | Restructure `_run_boltz_pass()` into three phases, replace per-structure `_run_boltz_for_structure()` calls with single directory invocation. `_run_boltz_for_structure()` is removed. |
| `tests/test_datamodule.py` | Update tests that patch `_run_boltz_for_structure` to patch `run_boltz_predict_dir` on the `boltz_runner` module instead |

## Files Unchanged

- `scripts/preprocess_monomers_frac_010.sh` — no changes needed
- `configs/data/dataset_monomers_len_128_frac_010.yaml` — no changes needed
- `src/quality_graft/data/cif_utils.py` — reused as-is
- `src/quality_graft/data/plddt_utils.py` — reused as-is
- `src/quality_graft/data/boltz_runner.py: build_boltz_command()` — body unchanged (parameter named `yaml_path` works with directory paths via `str()` cast)
