#!/bin/bash
#SBATCH --job-name=qg_train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err

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

mkdir -p "$PROJECT_ROOT/logs"

# --- Stage data to node-local scratch ---
DATA_SRC="/mnt/labs/data/bronstein/monomers_max_len_128_plddt_boltz"
DATA_DIR="/scratch/schekmenev/monomers_max_len_128_plddt_boltz"
NUM_DEVICES=2
DEVICE_TYPE=rtx6000

# ── NCCL tuning (critical for multi-GPU stability) ────────────────
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1          # Force SHM instead of P2P 
export NCCL_SHM_DISABLE=0          # Ensure shared memory transport is enabled

echo "Data source: $DATA_SRC"
echo "Data target: $DATA_DIR"
echo "GPUs: ${NUM_DEVICES}x $DEVICE_TYPE"

mkdir -p "$DATA_DIR"
echo "Staging data to scratch..."
cp -r "$DATA_SRC/processed" "$DATA_DIR/"
cp "$DATA_SRC"/*.csv "$DATA_DIR/" 2>/dev/null || true
N_PT=$(find "$DATA_DIR/processed" -name "*.pt" | wc -l)
echo "Staged $N_PT .pt files to $DATA_DIR/processed"
if [ "$N_PT" -eq 0 ]; then
    echo "ERROR: No .pt files staged. Aborting."
    exit 1
fi

# --- Run training ---
# CUDA_LAUNCH_BLOCKING=1 gives exact error location for CUDA asserts
export CUDA_LAUNCH_BLOCKING=1

echo "Starting training..."
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
    training.max_epochs=250 \
    training.check_val_every_n_epoch=10

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
