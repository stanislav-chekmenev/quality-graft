# W&B Logging for Preprocess Mode

**Date**: 2026-03-10
**Status**: Approved

## Problem

The `mode=preprocess` path in `train.py` runs dataset creation (PDB download, PyG conversion, Boltz-1 pLDDT labeling) but produces no W&B telemetry. Rich dataset metrics already exist in `wandb_logger.py` but are unused — they were written for an older argparse-based pipeline.

## Design

Hybrid approach: `train.py` owns the W&B run lifecycle; `datamodule.py` logs per-protein metrics live during the Boltz pass; `train.py` triggers the full dataset summary after preprocessing completes.

### Data Flow

```
train.py (preprocess mode)
  │
  ├── wandb.init(job_type="preprocessing", ...)
  │
  ├── dm.prepare_data()
  │     │
  │     ├── Pass 1: PDB filtering, download, PyG conversion (parent class)
  │     │
  │     └── Pass 2: _run_boltz_pass()
  │           │
  │           └── For each structure:
  │                 ├── time the Boltz prediction
  │                 ├── save pLDDT labels to .pt file
  │                 └── log_protein_metrics()  ← live W&B log (no-op if wandb.run is None)
  │
  ├── collect_dataset_stats(processed_dir)  ← scan ALL labeled .pt files
  │
  ├── log_dataset_summary(protein_stats)    ← plots + table covering full dataset
  │
  └── wandb.finish()
```

### Changes by File

#### `scripts/train.py`

In the `mode == "preprocess"` block:

1. Import `wandb` and the logging functions from `wandb_logger`
2. Call `wandb.init()` with:
   - `project` / `entity` / `name` from `cfg.training.wandb`
   - `job_type="preprocessing"`
   - `config=OmegaConf.to_container(cfg, resolve=True)`
3. After `dm.prepare_data()`, call `collect_dataset_stats()` then `log_dataset_summary()`
4. Call `wandb.finish()`

#### `src/quality_graft/data/datamodule.py`

In `_run_boltz_pass`:

1. Add `import time`
2. Wrap each `_run_boltz_for_structure` call with `time.time()` before/after
3. After successfully saving pLDDT labels, call `log_protein_metrics(structure_id, plddt_np, n_residues, elapsed_s, n_processed, n_failed, n_skipped)`
4. Import `log_protein_metrics` from `quality_graft.data.wandb_logger`

#### `src/quality_graft/data/wandb_logger.py`

1. Add `collect_dataset_stats(processed_dir: Path, num_plddt_bins: int = 50) -> list[dict]`:
   - Scan all `.pt` files in `processed_dir`
   - For each file with a `plddt` attribute, call `compute_protein_metrics()` with `elapsed_s=0.0`
   - Store raw pLDDT array in `_plddt_array` key (needed by `log_dataset_summary`)
   - Return the list of metric dicts
2. Remove `init_wandb_run()` (argparse-based, replaced by direct `wandb.init()` in `train.py`)
3. Remove `finish_wandb_run()` (replaced by direct `wandb.finish()` in `train.py`)

### W&B Run Structure

- **Project**: Same as training (`quality-graft` by default)
- **Job type**: `"preprocessing"` (distinguishes from `"training"` runs)
- **Config**: Full Hydra config is logged
- **Per-step logs**: `protein/*` metrics + `progress/*` counters (live during Boltz pass)
- **Summary logs**: 15+ plots across 5 categories + W&B Table with all proteins

### Guard Behavior

All W&B calls in `wandb_logger.py` are guarded by `wandb.run is not None`. If W&B init fails or is skipped, preprocessing still works — metrics are simply not logged.
