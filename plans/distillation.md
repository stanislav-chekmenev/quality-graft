# Confidence Head Distillation: Problem Statement & Implementation Plan

## Background

Quality-Graft trains a bridge between La-Proteina (a protein structure model) and Boltz1's confidence head to predict per-residue pLDDT scores from La-Proteina features. The current architecture is:

```
La-Proteina (frozen) → Adaptor (~1.2M params, trainable) → Boltz1 Confidence Head (frozen, 152.7M params) → pLDDT
```

The adaptor projects La-Proteina representations (768→384 single, 256→128 pair) and feeds them into the frozen Boltz1 confidence head, which contains a **48-block pairformer stack** (147.4M params) followed by linear prediction heads.

## The Overfitting Problem

Training runs at three dataset scales show severe overfitting:

| Experiment | Train loss | Val loss trend | Val accuracy (50 bins, random=2%) |
|---|---|---|---|
| 1 protein (train=val) | ~1.1 | 3.4 → 1.15 (decreasing) | 92% |
| 600 proteins | ~3.3 | 3.53 → 3.26 → 3.41 (increases) | ~8% |
| 5500 proteins | ~2.4 | 3.13 → 3.83 (steadily increases) | ~12% |

The model perfectly memorizes a single protein but shows essentially zero generalization on held-out proteins, even with 5500 training examples.

### Root Cause: The Frozen Pairformer Bottleneck

The 48-block frozen pairformer amplifies distributional mismatch between train and val. It was trained to process Boltz's own features and learned specific attention patterns and feature correlations from Boltz's pipeline. When the adaptor feeds it La-Proteina-derived features:

- **On train proteins**: the adaptor memorizes protein-specific feature patterns that happen to survive the 48 frozen pairformer layers
- **On val proteins**: slightly different features get progressively corrupted through each frozen pairformer layer — small errors compound through 48 layers of attention + triangle updates

The adaptor has to do "pre-distortion" — produce features that, after being processed by the frozen pairformer, yield correct outputs. This is much harder than learning a direct mapping, and it doesn't generalize.

## The Solution: Distill Into a Standalone Student Network

**Remove the frozen Boltz confidence head entirely.** Replace it with a smaller, fully trainable student network that takes La-Proteina features and directly predicts pLDDT.

### New architecture:

```
La-Proteina (frozen)
    → Projection layers (776→384 single, 256→128 pair)   [trainable, ~330K params]
    → Student Pairformer (4 blocks, trainable)             [NEW, ~12-13M params]
    → Linear pLDDT head (384→50)                           [NEW, trainable]
```

**Target parameter budget: 10-15M total** (distilling 152.7M → ~13M, a ~10x compression).

### Why this should work

1. **No frozen bottleneck** — all parameters get useful gradients, standard regularization (dropout, weight decay) works
2. **Student learns its own feature processing** — attention patterns develop for La-Proteina features rather than expecting Boltz-specific correlations
3. **Knowledge distillation with soft targets** — using Boltz's predicted pLDDT distributions (soft logits) as targets provides much richer signal per example than hard labels

## Existing Codebase

### Repository structure (relevant parts)

```
quality-graft/
├── configs/
│   ├── config.yaml                    # Top-level Hydra config
│   ├── model/
│   │   ├── adaptor.yaml               # Current adaptor config
│   │   ├── confidence_head.yaml       # Boltz confidence head config (to be replaced)
│   │   ├── quality_graft.yaml         # Full model assembly config
│   │   └── la_proteina_wrapper.yaml   # La-Proteina config
│   ├── data/
│   │   └── dataset.yaml               # Dataset config
│   └── training/
│       └── default.yaml               # Training hyperparameters
├── src/
│   ├── boltz/                         # Vendored Boltz1 subset (confidence module + layers)
│   │   └── model/
│   │       ├── layers/                # Transformer building blocks (reusable)
│   │       │   ├── attention.py       # AttentionPairBias
│   │       │   ├── dropout.py         # get_dropout_mask
│   │       │   ├── transition.py      # Transition (FFN)
│   │       │   ├── triangular_mult.py # TriangleMultiplication{Incoming,Outgoing}
│   │       │   ├── triangular_attention/attention.py  # TriangleAttention{Starting,Ending}Node
│   │       │   └── outer_product_mean.py
│   │       └── modules/
│   │           ├── trunk.py           # PairformerModule, PairformerLayer (reference architecture)
│   │           └── confidence.py      # ConfidenceModule, ConfidenceHeads
│   ├── la_proteina/                   # Vendored La-Proteina
│   └── quality_graft/
│       ├── models/
│       │   ├── adaptor.py             # AdaptorModule (current, to be extended)
│       │   ├── confidence_head.py     # BoltzConfidenceHead wrapper (to be replaced)
│       │   ├── la_proteina_wrapper.py # LaProteinaWrapper (keep as-is)
│       │   └── quality_graft.py       # QualityGraft assembly (needs new variant)
│       ├── data/
│       │   ├── datamodule.py          # QualityGraftDataModule (keep as-is)
│       │   └── plddt_utils.py         # plddt_to_bin, bin_to_plddt
│       └── training/
│           ├── lightning_module.py     # Training loop (needs new variant)
│           └── metrics.py             # plddt_accuracy, plddt_mae, pearson_r, spearman_r
```

### What exists and should be reused

**La-Proteina wrapper** (`src/quality_graft/models/la_proteina_wrapper.py`): Frozen. Extracts trunk_seqs [b,n,768], trunk_pair [b,n,n,256], local_latents [b,n,8], ca_coords [b,n,3]. Keep as-is.

**Current adaptor projections** (`src/quality_graft/models/adaptor.py`): The `single_proj` (LayerNorm + Linear 776→384) and `pair_proj` (LayerNorm + Linear 256→128) and `_binned_ca_distogram` are reusable. The `AdaptorAttentionBlock` class exists but is a simplified block — the student needs full pairformer blocks instead.

**Data pipeline** (`src/quality_graft/data/datamodule.py`): `QualityGraftDataModule` extends La-Proteina's `PDBLightningDataModule`. It runs Boltz-1 inference to generate pLDDT labels stored as `plddt_bin` (50-bin indices) in .pt files. The pLDDT targets come from Boltz's own predictions on each PDB structure (not crystallographic B-factors). Keep as-is.

**Metrics** (`src/quality_graft/training/metrics.py`): plddt_accuracy (top-1 bin match), plddt_mae, pearson_r, spearman_r. All masked. Keep as-is.

**Loss**: Currently cross-entropy on 50 pLDDT bins with padding mask (ignore_index=-1). See `QualityGraftLightningModule._compute_loss()`.

**Training config** (`configs/training/default.yaml`):
- AdamW: lr=1e-4, weight_decay=1e-2, betas=[0.9, 0.999]
- Linear warmup (200 steps) then linear decay to 1e-6
- batch_size=16, gradient_clip=1.0, bf16 precision
- Early stopping on val/plddt_accuracy, patience=5

### Boltz PairformerLayer architecture (reference for the student)

Each `PairformerLayer` in `src/boltz/model/modules/trunk.py` (lines 557-653) contains:

```python
# Pair track (z: [b, n, n, 128]):
z += dropout * TriangleMultiplicationOutgoing(z, pair_mask)
z += dropout * TriangleMultiplicationIncoming(z, pair_mask)
z += dropout * TriangleAttentionStartingNode(z, pair_mask)
z += dropout * TriangleAttentionEndingNode(z, pair_mask)    # columnwise dropout
z += Transition(z)                                           # FFN: 128 → 512 → 128

# Single track (s: [b, n, 384]):
s += AttentionPairBias(s, z, mask)                           # 16 heads, pair-biased
s += Transition(s)                                           # FFN: 384 → 1536 → 384
```

The Boltz confidence head uses these with: token_s=384, token_z=128, num_heads=16, dropout=0.25, pairwise_head_width=32, pairwise_num_heads=4. All building blocks are importable from `boltz.model.layers.*`.

### Key dimension reference

| La-Proteina | Adaptor Output / Student Input | Student Internal |
|---|---|---|
| trunk_seqs: [b,n,768] | s: [b,n,384] | s: [b,n,384] |
| trunk_pair: [b,n,n,256] | z: [b,n,n,128] | z: [b,n,n,128] |
| local_latents: [b,n,8] | (concatenated into single_proj input) | — |
| ca_coords: [b,n,3] | (distogram added to z) | — |

## What to Build

### 1. Preprocessing: Capture Boltz pLDDT Logits

The current preprocessing runs `boltz predict` as a subprocess, which only saves the continuous pLDDT value (after `softmax(logits) @ bin_centers`). The raw 50-bin logit distribution is computed inside Boltz's `ConfidenceHeads.to_plddt_logits(s)` but is never written to disk. For distillation training, we need these soft targets.

**Current flow (discards logits):**
```
boltz predict (subprocess)
  → ConfidenceHeads.forward()
    → plddt_logits [N, 50]  (raw logits — DISCARDED by writer)
    → softmax(logits) @ bin_centers → plddt [N]  (continuous — saved to npz)
  → writer saves plddt_{id}_model_0.npz with key "plddt"

datamodule reads npz:
  → plddt [N] (continuous)
  → plddt_to_bin() → plddt_bin [N] (hard integer labels)
  → stores both in .pt file
```

**New flow (captures logits):**
```
custom Boltz runner (in-process, not subprocess)
  → loads Boltz model once, runs batch inference
  → captures plddt_logits [N, 50] from ConfidenceHeads output dict
  → also captures plddt [N] (continuous) as before
  → saves both to npz: {"plddt": [N], "plddt_logits": [N, 50]}

datamodule reads npz:
  → plddt [N] (continuous) — as before
  → plddt_to_bin() → plddt_bin [N] (hard integer labels) — as before
  → plddt_logits [N, 50] (soft targets) — NEW
  → stores all three in .pt file
```

#### 1a. Custom In-Process Boltz Runner

**New file: `src/quality_graft/data/boltz_logit_runner.py`**

Instead of calling `boltz predict` as a subprocess (which uses Boltz's writer that discards logits), write a custom runner that loads and runs the Boltz model in-process:

1. **Load the Boltz1 model** — use the pip-installed `boltz` library's model loading utilities. The model includes diffusion + confidence modules.
2. **Process each structure** — feed the CIF/YAML through Boltz's data pipeline and run inference.
3. **Intercept the confidence output** — after the confidence forward pass, capture `plddt_logits` from the output dict before it gets collapsed. The key code path in the pip Boltz is:
   - `ConfidenceHeads.forward()` computes `plddt_logits = self.to_plddt_logits(s)` → shape `[N, 50]`
   - Then collapses to `plddt = compute_aggregated_metric(plddt_logits)` → shape `[N]`
   - Both are returned in the output dict, but the writer only saves the continuous value
4. **Save both** to npz files in the same directory structure Boltz would use, so the existing `find_plddt_npz()` / `_find_boltz_output()` functions still work. The npz now contains both `"plddt"` (float32 [N]) and `"plddt_logits"` (float32 [N, 50]).

**Implementation approach — monkey-patch the Boltz writer:**

The cleanest approach that avoids reimplementing Boltz's full inference pipeline:

```python
# Pseudocode for boltz_logit_runner.py
import boltz.data.write.writer as boltz_writer

def _patched_write_predictions(predictions, output_dir, ...):
    """Monkey-patched writer that also saves plddt_logits."""
    # Call original writer for all standard outputs
    original_write(predictions, output_dir, ...)
    # Additionally save plddt_logits if present
    if "plddt_logits" in predictions:
        for model_idx in range(predictions["plddt_logits"].shape[0]):
            logits = predictions["plddt_logits"][model_idx]
            path = output_dir / f"plddt_logits_{structure_id}_model_{model_idx}.npz"
            np.savez_compressed(path, plddt_logits=logits.cpu().numpy())

def run_boltz_with_logits(input_dir, out_dir, **kwargs):
    """Run boltz predict with patched writer to also save logits."""
    # Patch the writer, run prediction, restore
    ...
```

Alternatively, if the pip Boltz model API allows direct inference without going through the CLI, use that. The monkey-patch approach is a fallback that works with any Boltz version.

**Another option — post-hoc logit extraction:**

If monkey-patching is fragile, a simpler two-pass approach:
1. Run `boltz predict` as a subprocess (existing code, unchanged)
2. After subprocess completes, load only the Boltz confidence model (from `boltz1_conf.ckpt`) and run it on the predicted structures to re-extract logits

This is cleaner but runs the confidence head twice. Since the confidence head is fast compared to diffusion (~2% of total runtime), this is acceptable.

#### 1b. Updated `BoltzResult` and `BoltzBatchResult`

**File: `src/quality_graft/data/boltz_runner.py`**

Extend the result dataclasses:

```python
@dataclass
class BoltzResult:
    structure_id: str
    plddt: np.ndarray | None          # [N] float, 0-1 scale (existing)
    plddt_logits: np.ndarray | None   # [N, 50] float, raw logits (NEW)
    confidence_json: dict | None
    success: bool
    error_msg: str | None
```

Update `run_boltz_predict_dir()` result collection to also look for and load `plddt_logits` npz files:

```python
# In result collection loop:
npz_path = find_plddt_npz(lookup_dir, sid)
plddt = np.load(npz_path)["plddt"]

logits_npz_path = find_plddt_logits_npz(lookup_dir, sid)  # NEW
plddt_logits = np.load(logits_npz_path)["plddt_logits"] if logits_npz_path else None
```

#### 1c. Updated Datamodule Merge

**File: `src/quality_graft/data/datamodule.py`**

In `_run_boltz_pass()`, after loading the Boltz results, also store logits in the .pt file:

```python
# Existing:
graph.plddt = torch.tensor(plddt_np, dtype=torch.float32)
graph.plddt_bin = plddt_to_bin(graph.plddt, num_bins=self.num_plddt_bins)

# NEW:
if boltz_result.plddt_logits is not None:
    graph.plddt_logits = torch.tensor(boltz_result.plddt_logits, dtype=torch.float32)
```

This makes the .pt files contain:

| Field | Shape | Description |
|---|---|---|
| `plddt` | `[N]` float32 | Continuous pLDDT in [0, 1] (existing) |
| `plddt_bin` | `[N]` int64 | Hard bin labels [0, 49] (existing) |
| `plddt_logits` | `[N, 50]` float32 | Raw Boltz logits before softmax (NEW) |

#### 1d. Backward Compatibility

The pipeline must handle .pt files that were preprocessed without logits (existing data). During training, if `plddt_logits` is not present in a batch, fall back to hard-target cross-entropy loss only. This is handled via a simple `hasattr` check in the collate/batch step and the lightning module.

The `plddt_status.csv` does not need a new column — the presence of `plddt_logits` in the .pt file is checked at load time. Structures with `has_plddt=true` but no logits just use hard targets.

#### 1e. Re-preprocessing Existing Data

For existing preprocessed datasets, provide a standalone script to extract logits without re-running diffusion:

**New file: `scripts/extract_boltz_logits.py`**

```
python scripts/extract_boltz_logits.py \
    --data_dir data/pdb/ \
    --boltz_ckpt ckpt/boltz1_conf.ckpt \
    --devices 1
```

This script:
1. Scans `processed/*.pt` for files that have `plddt_bin` but not `plddt_logits`
2. For each, loads the structure, runs just the Boltz confidence head to get logits
3. Merges `plddt_logits` into the .pt file

Note: This requires the full Boltz model (not just the confidence head) to reproduce the exact same logits that the original `boltz predict` would have produced. The confidence head operates on features from the diffusion model, so we need to re-run the full pipeline or find the intermediate features. **If re-running full inference is too expensive for existing data, the alternative is to re-preprocess from scratch using the new logit-capturing pipeline (step 1a).**

### 2. Student Confidence Module

A new module that replaces `BoltzConfidenceHead`. It should:

- Accept `s: [b,n,384]` and `z: [b,n,n,128]` and `mask: [b,n]` (same interface as current confidence head)
- Run them through N trainable pairformer blocks (reuse `PairformerLayer` from `boltz.model.modules.trunk`)
- Apply final LayerNorms
- Project s → pLDDT logits via a linear head (384 → 50)
- Optionally also predict PDE (z → 64) and resolved (s → 2) for multi-task regularization

**Architecture parameters to make configurable:**
- `num_blocks`: number of pairformer layers (target: 4, giving ~12-13M params)
- `token_s`: 384 (match adaptor output)
- `token_z`: 128 (match adaptor output)
- `num_heads`: 16 (match Boltz)
- `dropout`: 0.15-0.25 (important for regularization — the current adaptor has zero dropout)
- `num_plddt_bins`: 50

**Parameter budget estimate per pairformer block (~3M):**
- TriangleMultiplicationOutgoing(128): ~200K
- TriangleMultiplicationIncoming(128): ~200K
- TriangleAttentionStartingNode(128, head_width=32, num_heads=4): ~100K
- TriangleAttentionEndingNode(128, head_width=32, num_heads=4): ~100K
- Transition(128, 512): ~130K
- AttentionPairBias(384, 128, 16 heads): ~740K
- Transition(384, 1536): ~1.2M
- Total per block: ~2.7M
- 4 blocks ≈ 10.8M + projections (~330K) + head (~20K) ≈ **~11.2M params**
- 5 blocks ≈ ~13.7M params

### 3. Updated Model Assembly

A new variant of `QualityGraft` (or updated to support both modes) that wires:
```
La-Proteina (frozen) → Adaptor projections (trainable) → Student pairformer (trainable) → pLDDT head (trainable)
```

The adaptor's projection layers (`single_proj`, `pair_proj`, `_binned_ca_distogram`) should be kept. The `AdaptorAttentionBlock` layers (the n_attn_layers attention blocks) should be removed since the student pairformer replaces that functionality.

### 4. Updated Lightning Module

The existing `QualityGraftLightningModule` needs updates:

**Freeze handling:**
- `on_train_epoch_start`: no more freezing confidence head — the student is fully trainable
- All trainable parameters include both adaptor projections AND student pairformer + head

**Distillation loss (the main change):**

The loss function must support both hard targets (cross-entropy on `plddt_bin`) and soft targets (KL divergence on Boltz logits). The combined loss is:

```python
def _compute_loss(self, student_logits, plddt_labels, mask, teacher_logits=None):
    """Combined hard + soft distillation loss.

    Parameters
    ----------
    student_logits : [b, n, 50]  — student's raw logits
    plddt_labels : [b, n] long   — hard bin targets
    mask : [b, n] float          — 1=valid, 0=padding
    teacher_logits : [b, n, 50] or None — Boltz's raw logits (soft targets)
    """
    # Hard target: cross-entropy (always computed)
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, 50),
        plddt_labels.reshape(-1),
        reduction="none",
        ignore_index=-1,
    )
    ce_loss = (ce_loss.view_as(plddt_labels) * mask).sum() / mask.sum().clamp(min=1)

    if teacher_logits is None:
        return ce_loss

    # Soft target: KL divergence with temperature scaling
    T = self.distill_temperature  # e.g. 2.0
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl_loss = (kl_loss * mask).sum() / mask.sum().clamp(min=1)
    kl_loss = kl_loss * (T ** 2)  # scale by T² to match gradient magnitudes

    # Weighted combination
    alpha = self.distill_alpha  # e.g. 0.7 (weight on soft targets)
    return (1 - alpha) * ce_loss + alpha * kl_loss
```

**Key distillation hyperparameters (add to config):**
- `distill_alpha`: weight on soft KL loss vs hard CE loss (default 0.7 — soft targets carry more info)
- `distill_temperature`: softmax temperature for KL divergence (default 2.0 — softens the distribution to expose inter-class relationships)

When `teacher_logits` is None (backward-compatible with old .pt files that lack logits), the loss degrades to pure cross-entropy.

**Same metrics** (accuracy, MAE, Pearson r, Spearman r) — these always use hard labels for evaluation.

### 5. Updated Configs

- New `configs/model/student_head.yaml` for the student confidence module
- Updated `configs/model/adaptor.yaml` with `n_attn_layers: 0` (projections only, no attention blocks)
- New or updated `configs/model/quality_graft.yaml` variant
- New distillation training params in `configs/training/default.yaml`:
  ```yaml
  distill_alpha: 0.7      # weight on soft KL loss (0.0 = pure CE, 1.0 = pure KL)
  distill_temperature: 2.0 # softmax temperature for KL divergence
  ```

### 6. Training considerations

- **Dropout is critical** — the current adaptor has zero dropout. The student pairformer should use dropout=0.15-0.25 on triangle operations and attention.
- **All parameters trainable** — adaptor projections + student pairformer + linear heads. No frozen downstream components.
- **Learning rate** may need adjustment — with ~13M params (vs 1.2M before), consider lr=5e-5 to 1e-4.
- **Gradient clipping** already at 1.0 in config.
- **Distillation temperature** — start with T=2.0. Higher T spreads probability mass more evenly across bins, giving the student richer gradients. Sweep [1.0, 2.0, 4.0] if needed.
- **Alpha schedule** — optionally anneal alpha from 1.0 (pure KL early, when student is random) to 0.5 (balanced later). Start with fixed alpha=0.7.

### 7. Tests

Add tests for the new preprocessing and student module:
- **Preprocessing tests:**
  - Verify .pt files contain `plddt_logits` field with shape `[N, 50]` after preprocessing
  - Verify `plddt_logits` values are reasonable (finite, not all zeros)
  - Verify backward compatibility: loading .pt files without `plddt_logits` doesn't crash
  - Verify the distillation loss falls back to CE-only when `teacher_logits` is None
- **Student module tests:**
  - Forward pass shape verification (same interface as BoltzConfidenceHead.forward)
  - Parameter count within budget (~10-15M)
  - Gradient flow verification (all parameters receive gradients)
  - Integration test with full pipeline (La-Proteina → adaptor → student → loss)
- **Distillation loss tests:**
  - KL divergence is zero when student matches teacher exactly
  - Combined loss with alpha=0 equals pure CE, alpha=1 equals pure KL
  - Temperature scaling produces softer distributions at higher T

## Important Notes

- The Boltz building blocks (`AttentionPairBias`, `TriangleMultiplication*`, `TriangleAttention*`, `Transition`) are all importable from `src/boltz/model/layers/` and should be reused directly.
- The `PairformerModule` and `PairformerLayer` classes in `src/boltz/model/modules/trunk.py` can potentially be reused directly with different `num_blocks`, or you can write a new module using the same layer building blocks.
- The `BoltzConfidenceHead` loads a 3.6GB checkpoint and instantiates the full 152.7M param module. The student head should NOT load any Boltz weights — it trains from scratch.
- The existing `ckpt/boltz1_conf.ckpt` checkpoint is still needed for La-Proteina wrapper but NOT for the student head.
- The Boltz pip package's `ConfidenceHeads.forward()` returns both `plddt_logits` and `plddt` in its output dict, but the writer (`boltz/data/write/writer.py`) only saves the continuous `plddt`. The preprocessing changes intercept this to capture the logits.
- Storage impact: adding `plddt_logits` [N, 50] float32 per structure adds ~200 bytes per residue (50 × 4 bytes). For a 200-residue protein this is ~40KB — negligible compared to the existing .pt file size.
- Read the existing code before modifying. The codebase has clear patterns — follow them.
