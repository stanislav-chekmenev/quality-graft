#!/bin/bash
#SBATCH --job-name=make_full_ds
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/preprocess_full_%j.out
#SBATCH --error=logs/preprocess_full_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
eval "$(conda shell.bash hook)" && conda activate quality_graft_env

PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

# --- Num GPUS ---
NUM_DEVICES=2

mkdir -p "$PROJECT_ROOT/logs"

# RTX6000 jobs: use /scratch for fast I/O
SCRATCH_DIR="/scratch/schekmenev/monomers_max_len_128_plddt_boltz_full"
mkdir -p "$SCRATCH_DIR"

echo "Data directory: $SCRATCH_DIR"

# Stage model weights to fast scratch (for follow-up train jobs)
CKPT_SCRATCH="/scratch/schekmenev/ckpt"
mkdir -p "$CKPT_SCRATCH"
echo "Staging model weights to $CKPT_SCRATCH..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_SCRATCH/" 2>/dev/null || echo "No ckpt files to stage (ok for preprocess-only)"
echo "Weight staging complete."

# ── NCCL tuning (critical for multi-GPU stability) ────────────────
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1          # Force SHM instead of P2P 
export NCCL_SHM_DISABLE=0          # Ensure shared memory transport is enabled

# Hydra debug
export HYDRA_FULL_ERROR=1         # Show full stack traces for Hydra errors

# --- Run full preprocessing ---
echo "Starting full preprocessing..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.local_only=false \
    data.data_dir="$SCRATCH_DIR" \
    data.selector_num_workers=16 \
    data.boltz.num_devices=$NUM_DEVICES \
    data.boltz.num_workers=4 \
    data.boltz.preprocessing_threads=8 \
    data.boltz.timeout_per_structure=180 \
    training.max_length=128

# Copy results to my data dir in the Bronstein group
echo "Copying results to group data directory..."
GROUP_DATA_DIR="/mnt/labs/data/bronstein"

mkdir -p "$GROUP_DATA_DIR"
cp -r "$SCRATCH_DIR" "$GROUP_DATA_DIR/"

echo "==========================================="
echo "Job completed: $(date)"
echo "Data at: $GROUP_DATA_DIR"
echo "==========================================="

