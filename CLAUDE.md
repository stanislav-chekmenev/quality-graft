# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quality-Graft trains an attention adaptor that bridges La-Proteina's protein structure representations to Boltz1's confidence head, enabling pLDDT prediction from La-Proteina features. The pipeline is: **La-Proteina (frozen) → Adaptor (trainable) → Boltz1 Confidence Head (frozen) → pLDDT scores**.

Two modes are supported:
- **Option A ("trunk")**: Trunk representations only (seqs + pair_rep + local_latents)
- **Option C ("hybrid")**: Trunk + decoder representations with gated fusion

## Environment Setup

UV Venv environment managed through pyproject.toml:
```bash
source .venv/bin/activate
```

PYTHONPATH must include both project root and src/:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/schekmenev/code_projects/quality-graft:/home/schekmenev/code_projects/quality-graft/src"
```

## Running Tests

```bash
# Unit tests only (no GPU/checkpoints needed)
pytest tests/ -v

# Include heavy tests (require GPU + checkpoints in ckpt/)
pytest tests/ -v --run-heavy

# Single test file
pytest tests/test_model_assembly.py -v

# Single test
pytest tests/test_model_assembly.py::TestQualityGraftUnit::test_forward_output_shapes -v
```

Tests use the `heavy` marker for tests requiring real checkpoints (ckpt/boltz1_conf.ckpt, ckpt/LD1_ucond_notri_512.ckpt, ckpt/AE1_ucond_512.ckpt). These are skipped by default; pass `--run-heavy` to include them.

## Configuration

Hydra configs in `configs/`. Top-level `configs/config.yaml` composes:
- `model/quality_graft.yaml` — full model assembly (adaptor via `_target_`, wrapper/head via factory methods)
- `model/adaptor.yaml` — adaptor architecture (source_mode, dims, attention layers)
- `model/la_proteina_wrapper.yaml` — checkpoint paths, decoder toggle
- `model/confidence_head.yaml` — Boltz1 architecture params
- `data/dataset.yaml`, `data/preprocessing.yaml`, `training/default.yaml`

## Architecture

### Source Code Layout (`src/`)

- **`src/quality_graft/`** — Project code
  - `models/quality_graft.py` — `QualityGraft(nn.Module)` assembles the full pipeline via dependency injection
  - `models/adaptor.py` — `AdaptorModule` projects La-Proteina dims (768→384 single, 256→128 pair) with optional self-attention and C-alpha distogram
  - `models/confidence_head.py` — `BoltzConfidenceHead` wraps Boltz1 `ConfidenceModule` with custom forward that bypasses input embedding and calls pairformer + linear heads directly
  - `models/la_proteina_wrapper.py` — `LaProteinaWrapper` wraps autoencoder + trunk + optional decoder; replicates forward passes to expose intermediate representations (seqs, pair_rep, local_latents)

- **`src/boltz/`** — Vendored minimal subset of Boltz1 (confidence module + dependencies only, ~25 files)
- **`src/la_proteina/`** — Vendored from NVIDIA fork (proteinfoundation/, openfold/, configs/)

### Key Design Decisions

- La-Proteina wrapper replicates forward passes rather than using hooks/subclassing to expose intermediate tensors, keeping vendored code unmodified
- Confidence head bypasses Boltz1 input embedding pipeline entirely — adaptor produces `s` and `z` directly in Boltz1 representation space, injected at the pairformer entry point
- `ConfidenceHeads.forward()` is also bypassed; individual linear heads (`to_plddt_logits`, `to_pde_logits`, `to_resolved_logits`) are called directly since aggregate metrics need Boltz-native features
- MSA module is instantiated for checkpoint compatibility but never called
- Option A→C transition uses zero-initialized `decoder_fusion` so the model starts identical to Option A

### Dimension Reference

| La-Proteina | Boltz1 Target |
|---|---|
| seqs: 768 | token_s: 384 |
| pair_rep: 256 | token_z: 128 |
| local_latents: 8 | — |
| Adaptor single input: 768+8=776 | → 384 |

## Detailed Architecture

See `plans/architecture.md` for full details including checkpoint structure, data flow diagrams, and bypass rationale.
