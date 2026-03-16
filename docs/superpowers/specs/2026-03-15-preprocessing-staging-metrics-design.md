# Multi-GPU Preprocessing, Weight Staging & Metric Checkpointing

Date: 2026-03-15

## Problem

1. Preprocessing is slow — runs on 1 GPU, Boltz workers share it serially
2. Model weights load from `/mnt/labs/home/` (slow networked storage)
3. Checkpointing uses `val/loss` but the metric of interest (`val/plddt_accuracy`) diverges from loss after ~20 epochs

## Changes

### 1. Multi-GPU Preprocessing

**Scope**: `_run_boltz_pass()` in `datamodule.py`, `preprocess_full.sbatch`, `preprocess_debug.sbatch`

- Request 4 GPUs in sbatch (`--gres=gpu:4`)
- Add config field `data.boltz.num_devices` (default 1)
- In `_run_boltz_pass()`, assign each ThreadPoolExecutor worker a GPU via round-robin: worker `i` sets `CUDA_VISIBLE_DEVICES=str(i % num_devices)` in the subprocess environment
- 5 workers per GPU (20 total with 4 GPUs)

**Data flow**: Same as today but workers run on different GPUs instead of contending on GPU 0.

### 2. Weight Staging to Fast Scratch

**Scope**: `preprocess_full.sbatch`, `preprocess_debug.sbatch`, `train_full.sbatch`

- Add a staging step in each sbatch script that copies `ckpt/*.ckpt` to scratch:
  - H100 nodes: `/netscratch/schekmenev/ckpt/`
  - RTX6000 nodes: `/scratch/schekmenev/ckpt/`
- Pass staged paths as Hydra overrides to `scripts/train.py`
- No Python code changes needed for this part

### 3. Checkpoint on pLDDT Accuracy

**Scope**: `scripts/train.py` (ModelCheckpoint + EarlyStopping)

- Change `ModelCheckpoint` to monitor `val/plddt_accuracy` with `mode="max"`
- Add `EarlyStopping` callback on `val/plddt_accuracy`, patience=5 (= 50 epochs at check_val_every_n_epoch=10)

## Files Modified

| File | Change |
|---|---|
| `src/quality_graft/data/datamodule.py` | GPU pinning in `_run_boltz_pass()` |
| `configs/data/dataset.yaml` | Add `boltz.num_devices` field |
| `scripts/preprocess_full.sbatch` | 4 GPUs, weight staging, Hydra overrides |
| `scripts/preprocess_debug.sbatch` | Multi-GPU option, weight staging |
| `scripts/train_full.sbatch` | Weight staging step + Hydra overrides |
| `scripts/train.py` | ModelCheckpoint monitor + EarlyStopping |

## Not Changed

- No changes to the model architecture or loss function
- No changes to the La-Proteina or Boltz vendored code
- Data staging for training (already exists in `train_full.sbatch`)
