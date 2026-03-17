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

### 1. Student Confidence Module

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

### 2. Updated Model Assembly

A new variant of `QualityGraft` (or updated to support both modes) that wires:
```
La-Proteina (frozen) → Adaptor projections (trainable) → Student pairformer (trainable) → pLDDT head (trainable)
```

The adaptor's projection layers (`single_proj`, `pair_proj`, `_binned_ca_distogram`) should be kept. The `AdaptorAttentionBlock` layers (the n_attn_layers attention blocks) should be removed since the student pairformer replaces that functionality.

### 3. Updated Lightning Module

The existing `QualityGraftLightningModule` needs updates:
- `on_train_epoch_start`: no more freezing confidence head — the student is fully trainable
- All trainable parameters include both adaptor projections AND student pairformer + head
- Loss remains cross-entropy on 50 pLDDT bins (soft target KL divergence is a future improvement)
- Same metrics (accuracy, MAE, Pearson r, Spearman r)

### 4. Updated Configs

- New `configs/model/student_head.yaml` for the student confidence module
- Updated `configs/model/adaptor.yaml` with `n_attn_layers: 0` (projections only, no attention blocks)
- New or updated `configs/model/quality_graft.yaml` variant

### 5. Training considerations

- **Dropout is critical** — the current adaptor has zero dropout. The student pairformer should use dropout=0.15-0.25 on triangle operations and attention.
- **All parameters trainable** — adaptor projections + student pairformer + linear heads. No frozen downstream components.
- **Learning rate** may need adjustment — with ~13M params (vs 1.2M before), consider lr=5e-5 to 1e-4.
- **Gradient clipping** already at 1.0 in config.

### 6. Tests

Add tests for the student module:
- Forward pass shape verification (same interface as BoltzConfidenceHead.forward)
- Parameter count within budget (~10-15M)
- Gradient flow verification (all parameters receive gradients)
- Integration test with full pipeline (La-Proteina → adaptor → student → loss)

## Important Notes

- The Boltz building blocks (`AttentionPairBias`, `TriangleMultiplication*`, `TriangleAttention*`, `Transition`) are all importable from `src/boltz/model/layers/` and should be reused directly.
- The `PairformerModule` and `PairformerLayer` classes in `src/boltz/model/modules/trunk.py` can potentially be reused directly with different `num_blocks`, or you can write a new module using the same layer building blocks.
- The `BoltzConfidenceHead` loads a 3.6GB checkpoint and instantiates the full 152.7M param module. The student head should NOT load any Boltz weights — it trains from scratch.
- The existing `ckpt/boltz1_conf.ckpt` checkpoint is still needed for La-Proteina wrapper but NOT for the student head.
- Read the existing code before modifying. The codebase has clear patterns — follow them.
