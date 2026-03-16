#!/usr/bin/env bash
# Debug preprocessing run using local CIF files (data/*.cif).
# Sets up directory structure, copies CIFs into raw/, and runs
# the preprocess pipeline with local_only=true (skips PDBManager).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEBUG_DIR="$PROJECT_ROOT/data/debug"

# --- Environment ---
# eval "$(micromamba shell hook --shell bash)" && micromamba activate quality_graft_env
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

# --- Prepare debug data directory ---
mkdir -p "$DEBUG_DIR/raw"
cp -n "$PROJECT_ROOT/data/1ubq.cif" "$DEBUG_DIR/raw/1ubq.cif" 2>/dev/null || true
cp -n "$PROJECT_ROOT/data/2gb1.cif" "$DEBUG_DIR/raw/2gb1.cif" 2>/dev/null || true

# --- Run preprocessing ---
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$DEBUG_DIR" \
    data.local_only=true \
    data.selector_num_workers=4 \
    data.boltz.chunk_size=1 \
    data.num_workers=4 \
    data.boltz.sampling_steps=20 \
    data.boltz.recycling_steps=1 \
    training.batch_size=1 \
    training.num_workers=1 \
    training.max_length=128
