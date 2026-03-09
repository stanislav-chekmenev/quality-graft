#!/usr/bin/env bash
# Debug training run: 2 epochs on 1 preprocessed structure (1ubq) on GPU.
# Assumes debug_preprocess.sh has already been run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEBUG_DIR="$PROJECT_ROOT/data/debug"

# --- Environment ---
eval "$(micromamba shell hook --shell bash)" && micromamba activate quality_graft_env
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

# --- Run training ---
python "$PROJECT_ROOT/scripts/train.py" \
    mode=train \
    data.data_dir="$DEBUG_DIR" \
    data.local_only=true \
    data.num_workers=0 \
    training.batch_size=1 \
    training.num_workers=0 \
    training.max_length=128 \
    training.max_epochs=2 \
    training.precision=32 \
    model.la_proteina_wrapper.device=cpu \
    model.confidence_head.device=cpu