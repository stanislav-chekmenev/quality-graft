#!/bin/bash
#SBATCH --job-name=make_full_ds
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
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

mkdir -p "$PROJECT_ROOT/logs"

# H100 jobs: use /netscratch for fast I/O
SCRATCH_DIR="/netscratch/schekmenev/monomers_max_len_128_plddt_boltz"
mkdir -p "$SCRATCH_DIR"

echo "Data directory: $SCRATCH_DIR"

# --- Run full preprocessing ---
echo "Starting full preprocessing..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=preprocess \
    data.data_dir="$SCRATCH_DIR" \
    data.selector_num_workers=32 \
    data.boltz.num_boltz_workers=20 \
    data.boltz.chunk_size=10 \
    training.max_length=128

# Copy results to my data dir in the Bronstein group
echo "Copying results to group data directory..."
GROUP_DATA_DIR="/mnt/labs/bronstein/schekmenev"

mkdir -p "$GROUP_DATA_DIR"
cp -r "$SCRATCH_DIR" "$GROUP_DATA_DIR/"

echo "==========================================="
echo "Job completed: $(date)"
echo "Data at: $GROUP_DATA_DIR"
echo "==========================================="

