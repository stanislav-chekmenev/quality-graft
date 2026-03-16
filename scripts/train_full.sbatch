#!/bin/bash
#SBATCH --job-name=qg_train
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"

NUM_DEVICES=2
DEVICE_TYPE=h100nvl

if [ "$DEVICE_TYPE" == "rtx6000" ]; then
    DATA_ROOT="/scratch/schekmenev"
elif [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/schekmenev"
fi

# Unpack .venv to scratch (once per node, skips if already present)
# Create tarball on login node:
#   uv venv --relocatable --python 3.11 && uv sync
#   tar czf "$PROJECT_ROOT/venv.tar.gz" -C "$PROJECT_ROOT" .venv
ENV_TARBALL="$PROJECT_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/.venv"
srun --ntasks-per-node=1 bash -c "
    if [ ! -d \"$ENV_LOCAL/bin\" ]; then
        mkdir -p \"$DATA_ROOT\"
        tar xzf \"$ENV_TARBALL\" -C \"$DATA_ROOT\"
        echo \"\$(hostname): env unpacked to $ENV_LOCAL\"
    else
        echo \"\$(hostname): env already at $ENV_LOCAL\"
    fi
"
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

mkdir -p "$PROJECT_ROOT/logs"

# --- Stage data to node-local scratch ---

GROUP_DATA_DIR="/mnt/labs/data/bronstein"
DIR_NAME="monomers_max_len_128_plddt_boltz_full"
DATA_SRC="$GROUP_DATA_DIR/$DIR_NAME"
DATA_DIR="$DATA_ROOT/$DIR_NAME"

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

# Stage model weights to node-local scratch
CKPT_DIR="$DATA_ROOT/ckpt"
mkdir -p "$CKPT_DIR"
echo "Staging model weights to $CKPT_DIR..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_DIR/"
echo "Weight staging complete."

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
    training.max_epochs=200 \
    training.check_val_every_n_epoch=10 \
    model.la_proteina_wrapper.proteina_ckpt_path="$CKPT_DIR/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$CKPT_DIR/AE1_ucond_512.ckpt" \
    model.confidence_head.ckpt_path="$CKPT_DIR/boltz1_conf.ckpt"

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="