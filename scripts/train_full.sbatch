#!/bin/bash
#SBATCH --job-name=boltz_distill
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
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
DEVICE_TYPE=rtx6000

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
REWRITE_ENV=true

echo "Setting up virtual environment on node-local scratch..."
srun --ntasks=1 bash -c "
    if [ ! -d \"$ENV_LOCAL/bin\" ] || [ "$REWRITE_ENV" = true ]; then
        mkdir -p \"$DATA_ROOT\"
        if ! tar xzf \"$ENV_TARBALL\" -C \"$DATA_ROOT\"; then
            echo \"ERROR: Failed to unpack $ENV_TARBALL\" >&2
            exit 1
        fi
        echo \"\$(hostname): env unpacked to $ENV_LOCAL\"
    else
        echo \"\$(hostname): env already at $ENV_LOCAL\"
    fi
"
if [ ! -d "$ENV_LOCAL/bin" ]; then
    echo "ERROR: Virtual environment not available at $ENV_LOCAL. Aborting."
    exit 1
fi
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"

echo "Virtual environment set up at $ENV_LOCAL"

# Ensure project src is importable
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# --- Stage data to node-local scratch ---
GROUP_DATA_DIR="/mnt/labs/data/bronstein"
DIR_NAME="monomers_max_len_128_plddt_boltz_full"
DATA_SRC="$GROUP_DATA_DIR/$DIR_NAME"
DATA_DIR="$DATA_ROOT/$DIR_NAME"

echo "Data source: $DATA_SRC"
echo "Data target: $DATA_DIR"
echo "GPUs: ${NUM_DEVICES}x $DEVICE_TYPE"

mkdir -p "$DATA_DIR"
echo "Staging data to scratch..."
STAGE_DATA=true

# Copy processed files if not already staged (assumes processed/ contains .pt files)
if [ -d "$DATA_DIR/processed" ] && \
    [ "$(find "$DATA_DIR/processed" -name "*.pt" | wc -l)" -gt 0 ] && \
    [ "$STAGE_DATA" = false ]; then
    echo "Data already staged at $DATA_DIR/processed, skipping copy."
else
    cp -r "$DATA_SRC/processed" "$DATA_DIR/"
    cp "$DATA_SRC"/*.csv "$DATA_DIR/" 2>/dev/null || true
fi

N_PT=$(find "$DATA_DIR/processed" -name "*.pt" | wc -l)
echo "Staged $N_PT .pt files to $DATA_DIR/processed"
if [ "$N_PT" -eq 0 ]; then
    echo "ERROR: No .pt files staged. Aborting."
    exit 1
fi

# Stage model weights to node-local scratch
WEIGHT_DIR="$DATA_ROOT/ckpt"
mkdir -p "$WEIGHT_DIR"
echo "Staging model weights to $WEIGHT_DIR..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$WEIGHT_DIR/"
echo "Weight staging complete."

# Training checkpoint directory (scratch, fast I/O)
TRAIN_CKPT_DIR="$DATA_ROOT/ckpt/runs"
mkdir -p "$TRAIN_CKPT_DIR"


# ── NCCL tuning (critical for multi-GPU stability) ────────────────
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1          # Force SHM instead of P2P 
export NCCL_SHM_DISABLE=0          # Ensure shared memory transport is enabled
export CUDA_LAUNCH_BLOCKING=1

# --- Run training ---
echo "Starting training..."
srun python "$PROJECT_ROOT/scripts/train.py" \
    mode=train \
    data.data_dir="$DATA_DIR" \
    data.local_only=true \
    data.batch_size=18 \
    data.num_workers=18 \
    training.distillation=true \
    training.max_length=128 \
    training.min_length=10 \
    training.accelerator=gpu \
    training.devices=$NUM_DEVICES \
    training.strategy=ddp \
    training.precision=bf16-mixed \
    training.max_epochs=100 \
    training.check_val_every_n_epoch=5 \
    training.checkpoint_dir="$TRAIN_CKPT_DIR" \
    model.adaptor.n_attn_layers=1 \
    model.la_proteina_wrapper.proteina_ckpt_path="$WEIGHT_DIR/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$WEIGHT_DIR/AE1_ucond_512.ckpt" \
    model/confidence_head=student

# Copy latest run checkpoints back to project root
LATEST_RUN=$(ls -1td "$TRAIN_CKPT_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    RUN_NAME=$(basename "$LATEST_RUN")
    echo "Copying run $RUN_NAME to $PROJECT_ROOT/ckpt/runs/$RUN_NAME..."
    mkdir -p "$PROJECT_ROOT/ckpt/runs"
    cp -rv "$LATEST_RUN" "$PROJECT_ROOT/ckpt/runs/"
else
    echo "No checkpoint runs found to copy."
fi


echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="