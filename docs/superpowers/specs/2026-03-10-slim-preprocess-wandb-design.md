# Slim Preprocessing W&B Logger

**Date**: 2026-03-10
**Status**: Approved

## Problem

W&B logging during preprocessing adds ~2 minutes per protein due to:
- Per-protein `wandb.log()` calls with 20+ metrics in the hot loop
- 15+ matplotlib plots generated at the end
- Full W&B Table with 19 columns

Additionally, loguru format strings use `%d`/`%s` (stdlib style) instead of `{}` (loguru style), causing placeholders to print literally.

## Design

### 1. Fix loguru format strings

**Files**: `src/quality_graft/data/datamodule.py`, `src/quality_graft/data/boltz_runner.py`

Replace all `%d`/`%s`/`%.3f` with `{}`/`{:.3f}` in loguru logger calls.

### 2. Remove per-protein W&B logging

**File**: `src/quality_graft/data/datamodule.py`

Delete the `log_protein_metrics()` call from `_run_boltz_pass()`. Remove the import.

### 3. Rewrite `wandb_logger.py`

**Delete** (no longer needed):
- `longest_contiguous_below()`
- `count_segments_below()`
- `compute_protein_metrics()`
- `log_protein_metrics()`
- All matplotlib plotting code (15 plots)
- W&B Table generation

**Keep/rewrite**:
- `collect_dataset_stats(processed_dir)` — scan .pt files, return list of `(structure_id, mean_plddt, n_residues)` tuples
- `log_dataset_summary(protein_stats)` — single `wandb.log()` call with:
  - `dataset/plddt_histogram`: `wandb.Histogram` of per-protein mean pLDDT values
  - `dataset/mean_plddt`: mean of per-protein means
  - `dataset/std_plddt`: std of per-protein means
  - `dataset/max_plddt`: max of per-protein means
  - `dataset/min_plddt`: min of per-protein means
  - `dataset/num_proteins`: count of labeled structures

### 4. W&B stays active for system metrics

**File**: `scripts/train.py`

No change to `wandb.init()` — it still runs at the start of preprocessing so W&B auto-collects CPU/GPU/memory stats. The single `wandb.log()` call at the end logs the dataset summary.

## Expected outcome

- Per-protein overhead: eliminated (no `wandb.log()` in hot loop)
- End-of-run logging: single call with 6 values + 1 histogram
- Loguru output: properly formatted with protein names, pLDDT values, residue counts
