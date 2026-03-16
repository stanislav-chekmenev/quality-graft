#!/bin/bash
#SBATCH --job-name=qg_prep_debug
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gres=gpu:h100nvl:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preprocess_debug_%j.out
#SBATCH --error=logs/preprocess_debug_%j.err

set -eo pipefail

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "==========================================="

# --- Environment ---
eval "$(conda shell.bash hook)" && conda activate quality_graft_env

PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

mkdir -p "$PROJECT_ROOT/logs"

# H100 jobs: use /netscratch for fast I/O
SCRATCH_DIR="/netscratch/schekmenev/qg_debug"
mkdir -p "$SCRATCH_DIR"

echo "Data directory: $SCRATCH_DIR"

# Stage model weights to fast scratch (for follow-up train jobs)
CKPT_SCRATCH="/netscratch/schekmenev/ckpt"
mkdir -p "$CKPT_SCRATCH"
echo "Staging model weights to $CKPT_SCRATCH..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_SCRATCH/" 2>/dev/null || echo "No ckpt files to stage (ok for preprocess-only)"
echo "Weight staging complete."

# --- Run preprocessing (tiny fraction for smoke test) ---
echo "Starting debug preprocessing..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$SCRATCH_DIR" \
    data.fraction=0.005 \
    data.selector_num_workers=8 \
    data.boltz.num_devices=4 \
    data.boltz.num_boltz_workers=8 \
    data.boltz.chunk_size=5 \
    data.boltz.sampling_steps=20 \
    data.boltz.recycling_steps=1 \
    training.batch_size=1 \
    training.num_workers=1 \
    training.max_length=128

# Copy results to my data dir in the Bronstein group
echo "Copying results to group data directory..."
GROUP_DATA_DIR="/mnt/labs/bronstein/schekmenev/qg_debug"

mkdir -p "$GROUP_DATA_DIR"
cp -r "$SCRATCH_DIR" "$GROUP_DATA_DIR/"

echo "==========================================="
echo "Job completed: $(date)"
echo "Data at: $GROUP_DATA_DIR"
echo "==========================================="
