# Training Pipeline Design

## Overview

Build the training pipeline for Quality-Graft: data preprocessing with integrated Boltz-1 pLDDT label generation, a Lightning training module, and a Hydra-configured training script.

---

## 1. Data Pipeline

### `QualityGraftDataModule` (subclass of `PDBLightningDataModule`)

**Location**: `src/quality_graft/data/datamodule.py`

Subclasses `PDBLightningDataModule` to add Boltz-1 pLDDT label generation while keeping the vendored La-Proteina code unmodified.

### `prepare_data()` — Two-Pass Approach

**Pass 1 (parent)**: Call `super().prepare_data()` — runs `PDBDataSelector` filtering (max_length=128, monomers, etc.), downloads CIF files, converts each to a PyG `Data` object via `protein_to_pyg()`, saves as `{pdb}_{chain}.pt` in `processed/`.

**Pass 2 (Boltz pLDDT)**: Load the Boltz-1 model once. Iterate over all `.pt` files in `processed/`. For each that doesn't already have a `plddt_bin` attribute:
- Find the corresponding CIF in `raw/`
- Run Boltz-1 prediction (reusing `boltz_runner.py` / `plddt_utils.py`)
- Add `graph.plddt` (float) and `graph.plddt_bin` (long, 50 bins) to the `Data` object
- Re-save the `.pt` file

The filtered structure list comes from the CSV saved by pass 1, so pass 2 knows exactly which structures to process.

### `setup()` and `_get_dataset()`

Inherited from parent. `PDBDataset.__getitem__()` returns the full PyG graph, which now includes `plddt_bin`. The `DensePaddingDataLoader` handles batching with padding.

---

## 2. Training Module

### `QualityGraftLightningModule` (wraps `QualityGraft`)

**Location**: `src/quality_graft/training/lightning_module.py`

Wraps the existing `QualityGraft(nn.Module)` — no changes to the model itself.

**Key design points:**
- Only adaptor parameters passed to optimizer (La-Proteina and confidence head stay frozen)
- Loss: cross-entropy on pLDDT bins (50 bins), masked by residue mask

### `training_step(batch)`

1. Extract `coords_nm`, `coord_mask`, `residue_type`, `mask` from PyG batch
2. Forward through `QualityGraft` -> `plddt_logits [b, n, 50]`
3. Get `plddt_bin` labels from batch
4. Compute masked cross-entropy loss
5. Log `train/loss` to W&B

### `validation_step(batch)`

Same forward pass, plus:
- `val/loss`, `val/plddt_accuracy`, `val/plddt_mae`, `val/pearson_r`, `val/spearman_r`
- Correlations computed per-protein, then averaged

### `configure_optimizers()`

- AdamW on `self.model.trainable_parameters()` only
- Linear LR scheduler with warmup
- LR, weight decay, warmup steps from Hydra config

---

## 3. Metrics

**Location**: `src/quality_graft/training/metrics.py`

Four validation metrics, all computed per-protein then averaged:

| Metric | Description |
|---|---|
| **pLDDT Accuracy** | Top-1 bin match between predicted argmax and ground truth `plddt_bin`, masked |
| **pLDDT MAE** | Convert predicted (expected value from softmax) and ground truth bins to continuous [0,1] via bin centers, compute mean absolute error |
| **Pearson R** | Correlation between predicted and ground truth continuous pLDDT per protein |
| **Spearman R** | Rank correlation between predicted and ground truth continuous pLDDT per protein |

---

## 4. Training Script

### `scripts/train.py`

Entry point with two modes:

- `--mode=preprocess`: Instantiates `QualityGraftDataModule`, calls `prepare_data()` only (both passes). No model loading needed.
- `--mode=train`: Instantiates full pipeline — data module, model, Lightning `Trainer` — calls `trainer.fit()`. Assumes preprocessing is done.

Uses Hydra to compose all configs.

---

## 5. Hydra Config

### `configs/training/default.yaml`

```yaml
optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.999]

scheduler:
  type: linear
  warmup_steps: 500
  min_lr: 1.0e-6

max_length: 128
batch_size: 4
num_workers: 4
precision: bf16
max_epochs: 50
gradient_clip_val: 1.0
accumulate_grad_batches: 1

wandb:
  project: quality-graft
  entity: null
  run_name: null
```

### `configs/data/dataset.yaml`

```yaml
data_dir: data/pdb/
max_length: ${training.max_length}
min_length: 10
molecule_type: protein
oligomeric_min: 1
oligomeric_max: 1
format: cif
num_plddt_bins: 50

boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 200
  recycling_steps: 3
  devices: 1
  accelerator: gpu
```

---

## 6. File Changes

### New files

| File | Purpose |
|---|---|
| `src/quality_graft/data/datamodule.py` | `QualityGraftDataModule` (subclass of `PDBLightningDataModule`) |
| `src/quality_graft/training/__init__.py` | Package init |
| `src/quality_graft/training/lightning_module.py` | `QualityGraftLightningModule` |
| `src/quality_graft/training/metrics.py` | pLDDT accuracy, MAE, Pearson, Spearman |
| `scripts/train.py` | Entry point (preprocess / train modes) |

### Modified files

| File | Change |
|---|---|
| `configs/training/default.yaml` | Populate with training config |
| `configs/data/dataset.yaml` | Add data module + Boltz config |
| `configs/data/preprocessing.yaml` | Remove TODO (handled by data module) |

### Unchanged

- All vendored code (`src/la_proteina/`, `src/boltz/`)
- `QualityGraft(nn.Module)`, `AdaptorModule`, `BoltzConfidenceHead`, `LaProteinaWrapper`
- Existing `src/quality_graft/data/` utilities (`boltz_runner.py`, `plddt_utils.py`, `cif_utils.py`) — reused by the data module