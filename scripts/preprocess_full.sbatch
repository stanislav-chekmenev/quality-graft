#!/bin/bash
#SBATCH --job-name=reprocess_boltz
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_full_%j.out
#SBATCH --error=logs/preprocess_full_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"

DEVICE_TYPE=rtx6000

if [ "$DEVICE_TYPE" == "rtx6000" ]; then
    DATA_ROOT="/scratch/schekmenev"
elif [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/schekmenev"
fi

# Unpack .venv to scratch (skips if already present)
# Create tarball on login node:
#   uv venv --relocatable --python 3.11 && uv sync
#   tar czf "$PROJECT_ROOT/venv.tar.gz" -C "$PROJECT_ROOT" .venv
ENV_TARBALL="$PROJECT_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/.venv"
REWRITE_ENV=true


if [ ! -d "$ENV_LOCAL/bin" ] || [ "$REWRITE_ENV" = true ]; then
    mkdir -p "$DATA_ROOT"
    if ! tar xzf "$ENV_TARBALL" -C "$DATA_ROOT"; then
        echo "ERROR: Failed to unpack $ENV_TARBALL" >&2
        exit 1
    fi
    echo "Env unpacked to $ENV_LOCAL"
else
    echo "Env already present at $ENV_LOCAL"
fi
if [ ! -d "$ENV_LOCAL/bin" ]; then
    echo "ERROR: Virtual environment not available at $ENV_LOCAL. Aborting."
    exit 1
fi
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"

# --- Num GPUS ---
NUM_DEVICES=1

mkdir -p "$PROJECT_ROOT/logs"

SCRATCH_DIR="$DATA_ROOT/monomers_max_len_128_plddt_boltz_full"
mkdir -p "$SCRATCH_DIR"
echo "Data directory: $SCRATCH_DIR"

# Stage model weights to fast scratch (for follow-up train jobs)
CKPT_SCRATCH="$DATA_ROOT/ckpt"
mkdir -p "$CKPT_SCRATCH"
echo "Staging model weights to $CKPT_SCRATCH..."
cp -v "$PROJECT_ROOT/ckpt/"*.ckpt "$CKPT_SCRATCH/" 2>/dev/null || echo "No ckpt files to stage (ok for preprocess-only)"
echo "Weight staging complete."

# ── NCCL tuning (critical for multi-GPU stability) ────────────────
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # Disable InfiniBand if not available
export NCCL_SOCKET_IFNAME=eth0     # Restrict NCCL to Ethernet interfaces 
export NCCL_P2P_DISABLE=1          # Force SHM instead of P2P 
export NCCL_SHM_DISABLE=0          # Ensure shared memory transport is enabled

# Hydra debug
export HYDRA_FULL_ERROR=1         # Show full stack traces for Hydra errors

# (Optional) Make PyTorch crash immediately on distributed errors with a stack trace
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Controls how long PyTorch waits for other processes to join the group
# Useful if one node fails early and the other is just hanging.
export TORCH_NCCL_BLOCKING_WAIT=1  # Forces synchronous error reporting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # Enable async error handling

# Controls how long NCCL waits for the initial handshake between nodes
# Default is 15-30 minutes. Set to 60 seconds for debugging.
export NCCL_COMM_ID_TIMEOUT=60

# --- Reprocess Boltz (re-run predictions, save logits into .pt files) ---
echo "Starting Boltz reprocessing..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.local_only=true \
    data.reprocess_boltz=true \
    data.data_dir="$SCRATCH_DIR" \
    data.boltz.num_devices=$NUM_DEVICES \
    data.boltz.chunk_size=5 \
    data.boltz.max_parallel_samples=5 \
    data.boltz.sampling_steps=50 \
    data.boltz.recycling_steps=1 \
    data.boltz.num_workers=4 \
    data.boltz.preprocessing_threads=12 \
    data.boltz.timeout_per_structure=120 \
    training.max_length=128 \
    training.min_length=10

# Copy results to my data dir in the Bronstein group
echo "Copying results to group data directory..."
GROUP_DATA_DIR="/mnt/labs/data/bronstein"

mkdir -p "$GROUP_DATA_DIR"
cp -r "$SCRATCH_DIR" "$GROUP_DATA_DIR/"

echo "==========================================="
echo "Job completed: $(date)"
echo "Data at: $GROUP_DATA_DIR"
echo "==========================================="

