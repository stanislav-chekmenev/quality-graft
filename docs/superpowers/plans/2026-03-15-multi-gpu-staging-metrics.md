# Multi-GPU Preprocessing, Weight Staging & Metric Checkpointing

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multi-GPU preprocessing with per-worker GPU pinning, stage model weights to fast scratch before jobs, and checkpoint on pLDDT accuracy instead of loss.

**Architecture:** Three independent changes: (1) GPU round-robin in `_run_boltz_pass` + `run_boltz_predict_dir`, (2) weight copy steps in sbatch scripts with Hydra overrides, (3) swap ModelCheckpoint monitor + add EarlyStopping.

**Tech Stack:** Python, PyTorch Lightning, SLURM sbatch, Hydra

---

## Chunk 1: Multi-GPU Preprocessing

### Task 1: Add `cuda_device` parameter to `run_boltz_predict_dir`

**Files:**
- Modify: `src/quality_graft/data/boltz_runner.py:288-349`

- [ ] **Step 1: Add `cuda_device` parameter to `run_boltz_predict_dir`**

Add `cuda_device: int | None = None` parameter. When set, inject `CUDA_VISIBLE_DEVICES=str(cuda_device)` into the subprocess environment after `_clean_env_for_boltz()`.

```python
# In run_boltz_predict_dir, after line 348 (env = _clean_env_for_boltz()):
        env = _clean_env_for_boltz()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
```

The function signature becomes:
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
    cuda_device: int | None = None,
) -> BoltzBatchResult:
```

- [ ] **Step 2: Verify no tests break**

Run: `pytest tests/ -v -x`
Expected: All pass (existing tests don't use `cuda_device`)

- [ ] **Step 3: Commit**

```bash
git add src/quality_graft/data/boltz_runner.py
git commit -m "Add cuda_device param to run_boltz_predict_dir"
```

### Task 2: Add `num_devices` config and GPU round-robin in `_run_boltz_pass`

**Files:**
- Modify: `configs/data/dataset.yaml:22-30`
- Modify: `src/quality_graft/data/datamodule.py:205-320`

- [ ] **Step 1: Add `num_devices` to boltz config**

In `configs/data/dataset.yaml`, add under the `boltz:` section:
```yaml
boltz:
  model: boltz1
  diffusion_samples: 1
  sampling_steps: 200
  recycling_steps: 3
  devices: 1
  accelerator: gpu
  use_msa_server: false
  timeout_per_structure: 300
  num_devices: 1  # Number of GPUs for parallel preprocessing
```

- [ ] **Step 2: Modify `_run_boltz_pass` to pin workers to GPUs**

In `datamodule.py`, read `num_devices` from config and pass `cuda_device` to each submitted chunk:

```python
def _run_boltz_pass(self, file_names: List[str]) -> None:
    num_boltz_workers = self.boltz_config.get("num_boltz_workers", 2)
    chunk_size = self.boltz_config.get("chunk_size", 10)
    num_devices = self.boltz_config.get("num_devices", 1)
```

Then in the submission loop (around line 309-319), assign each chunk a GPU via round-robin and pass it through:

```python
            future_to_chunk = {}
            for idx, (chunk_sids, inp_dir, out_dir) in enumerate(
                zip(valid_chunks, chunk_input_dirs, chunk_output_dirs)
            ):
                # Round-robin GPU assignment
                cuda_device = idx % num_devices if num_devices > 1 else None
                future = executor.submit(
                    run_boltz_predict_dir,
                    input_dir=inp_dir,
                    out_dir=out_dir,
                    structure_ids=chunk_sids,
                    cuda_device=cuda_device,
                    **boltz_kwargs,
                )
                future_to_chunk[future] = (idx, chunk_sids)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v -x`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add configs/data/dataset.yaml src/quality_graft/data/datamodule.py
git commit -m "Add multi-GPU round-robin for preprocessing workers"
```

### Task 3: Update sbatch scripts for multi-GPU preprocessing

**Files:**
- Modify: `scripts/preprocess_full.sbatch`
- Modify: `scripts/preprocess_debug.sbatch`

- [ ] **Step 1: Update `preprocess_full.sbatch`**

Change GPU request from 1 to 4 and set workers to 20 (5 per GPU):

```bash
#SBATCH --gres=gpu:h100nvl:4
```

Update the python command to include `data.boltz.num_devices=4` and `data.boltz.num_boltz_workers=20`:

```bash
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$SCRATCH_DIR" \
    data.selector_num_workers=32 \
    data.boltz.num_boltz_workers=20 \
    data.boltz.num_devices=4 \
    data.boltz.chunk_size=10 \
    data.boltz.timeout_per_structure=300 \
    training.max_length=128
```

- [ ] **Step 2: Update `preprocess_debug.sbatch`**

Change to 4 GPUs and adjust workers (keep smaller for debug but still multi-GPU):

```bash
#SBATCH --gres=gpu:h100nvl:4
```

Update python command:
```bash
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$SCRATCH_DIR" \
    data.fraction=0.005 \
    data.selector_num_workers=8 \
    data.boltz.num_boltz_workers=8 \
    data.boltz.num_devices=4 \
    data.boltz.chunk_size=5 \
    data.boltz.sampling_steps=20 \
    data.boltz.recycling_steps=1 \
    training.batch_size=1 \
    training.num_workers=1 \
    training.max_length=128
```

- [ ] **Step 3: Commit**

```bash
git add scripts/preprocess_full.sbatch scripts/preprocess_debug.sbatch
git commit -m "Request 4 GPUs for preprocessing sbatch jobs"
```

## Chunk 2: Weight Staging to Fast Scratch

### Task 4: Add weight staging to preprocessing sbatch scripts

**Files:**
- Modify: `scripts/preprocess_full.sbatch`
- Modify: `scripts/preprocess_debug.sbatch`

- [ ] **Step 1: Add weight staging to `preprocess_full.sbatch`**

After the `mkdir -p "$SCRATCH_DIR"` line, add checkpoint staging:

```bash
# Stage model weights to fast scratch
CKPT_SCRATCH="/netscratch/schekmenev/ckpt"
mkdir -p "$CKPT_SCRATCH"
echo "Staging model weights to $CKPT_SCRATCH..."
cp -v "$PROJECT_ROOT/ckpt/boltz1_conf.ckpt" "$CKPT_SCRATCH/" 2>/dev/null || true
cp -v "$PROJECT_ROOT/ckpt/LD1_ucond_notri_512.ckpt" "$CKPT_SCRATCH/" 2>/dev/null || true
cp -v "$PROJECT_ROOT/ckpt/AE1_ucond_512.ckpt" "$CKPT_SCRATCH/" 2>/dev/null || true
echo "Weight staging complete."
```

Then add Hydra overrides for checkpoint paths to the python command:

```bash
    model.la_proteina_wrapper.proteina_ckpt_path="$CKPT_SCRATCH/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$CKPT_SCRATCH/AE1_ucond_512.ckpt" \
    model.confidence_head.ckpt_path="$CKPT_SCRATCH/boltz1_conf.ckpt"
```

Note: preprocessing only uses Boltz (via pip subprocess), not La-Proteina or the confidence head. The ckpt overrides are only needed for `mode=train`. For preprocessing, omit the model overrides — they won't be loaded anyway. Only stage weights if you want them cached on scratch for a subsequent train job.

Actually, for preprocessing the model weights are NOT loaded (only Boltz subprocess runs, using pip-installed boltz). So skip the Hydra overrides in preprocess scripts — just stage the weights for convenience so a follow-up train job can use them.

```bash
# Stage model weights to fast scratch (for follow-up train jobs)
CKPT_SCRATCH="/netscratch/schekmenev/ckpt"
mkdir -p "$CKPT_SCRATCH"
echo "Staging model weights to $CKPT_SCRATCH..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_SCRATCH/" 2>/dev/null || echo "No ckpt files to stage (ok for preprocess-only)"
echo "Weight staging complete."
```

- [ ] **Step 2: Add weight staging to `preprocess_debug.sbatch`**

Same block as above (both use /netscratch since both are H100 partition).

- [ ] **Step 3: Commit**

```bash
git add scripts/preprocess_full.sbatch scripts/preprocess_debug.sbatch
git commit -m "Stage model weights to /netscratch in preprocess jobs"
```

### Task 5: Add weight staging to training sbatch script

**Files:**
- Modify: `scripts/train_full.sbatch`

- [ ] **Step 1: Add weight staging to `train_full.sbatch`**

After the existing data staging block (around line 48), add:

```bash
# Stage model weights to node-local scratch
CKPT_DIR="/scratch/schekmenev/ckpt"
mkdir -p "$CKPT_DIR"
echo "Staging model weights to $CKPT_DIR..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_DIR/"
echo "Weight staging complete."
```

Then add Hydra overrides to the srun command:

```bash
srun python "$PROJECT_ROOT/scripts/train.py" \
    mode=train \
    data.data_dir="$DATA_DIR" \
    data.local_only=true \
    data.batch_size=6 \
    data.num_workers=6 \
    training.max_length=128 \
    training.accelerator=gpu \
    training.devices=$NUM_DEVICES \
    training.strategy=ddp \
    training.precision=bf16-mixed \
    training.max_epochs=100 \
    training.check_val_every_n_epoch=10 \
    model.la_proteina_wrapper.proteina_ckpt_path="$CKPT_DIR/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$CKPT_DIR/AE1_ucond_512.ckpt" \
    model.confidence_head.ckpt_path="$CKPT_DIR/boltz1_conf.ckpt"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_full.sbatch
git commit -m "Stage model weights to /scratch in train job"
```

## Chunk 3: Metric Checkpointing

### Task 6: Change ModelCheckpoint to monitor pLDDT accuracy + add EarlyStopping

**Files:**
- Modify: `scripts/train.py:26,185-195`

- [ ] **Step 1: Update imports**

Add `EarlyStopping` to the import:

```python
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
```

- [ ] **Step 2: Change ModelCheckpoint and add EarlyStopping**

Replace the callbacks list (lines 185-195):

```python
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{train_cfg.checkpoint_dir}/{run_timestamp}",
            monitor="val/plddt_accuracy",
            mode="max",
            save_top_k=3,
            filename="epoch{epoch:02d}-val_acc{val/plddt_accuracy:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/plddt_accuracy",
            mode="max",
            patience=5,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v -x`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "Checkpoint on plddt_accuracy, add early stopping"
```
