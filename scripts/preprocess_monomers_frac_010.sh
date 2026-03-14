#!/usr/bin/env bash
# Preprocess ~500 high-quality monomeric proteins (max length 128) with pLDDT labels.
# Uses PDBDataSelector to download from PDB with fraction=0.05, resolution <= 2.5 A.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Environment ---
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

# --- Run preprocessing ---
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
