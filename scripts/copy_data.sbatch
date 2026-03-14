#!/bin/bash
#SBATCH --job-name=make_full_ds
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/preprocess_full_%j.out
#SBATCH --error=logs/preprocess_full_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

SCRATCH_DIR="/netscratch/schekmenev/monomers_max_len_128_plddt_boltz"
echo "Data directory: $SCRATCH_DIR"

# Copy results to my data dir in the Bronstein group
echo "Copying results to group data directory..."
GROUP_DATA_DIR="/mnt/labs/data/bronstein"

cp -r "$SCRATCH_DIR" "$GROUP_DATA_DIR/"

echo "==========================================="
echo "Job completed: $(date)"
echo "Data at: $GROUP_DATA_DIR"
echo "==========================================="

