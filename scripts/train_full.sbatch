#!/bin/bash
#SBATCH --job-name=qg_train
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
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

# Data on fast scratch (already preprocessed)
DATA_DIR="/netscratch/schekmenev/monomers_max_len_128_plddt_boltz"

echo "Data directory: $DATA_DIR"
echo "GPUs: 2x H100"

# --- Run training ---
echo "Starting training..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=train \
    data.data_dir="$DATA_DIR" \
    data.batch_size=96 \
    data.num_workers=16 \
    training.max_length=128 \
    training.accelerator=gpu \
    training.devices=2 \
    training.strategy=ddp \
    training.precision=bf16 \
    training.max_epochs=1500 \
    training.check_val_every_n_epoch=50

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
