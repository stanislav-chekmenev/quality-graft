#!/bin/bash
#SBATCH --job-name=qg_ffprobe
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ffprobe_%j.out
#SBATCH --error=logs/ffprobe_%j.err

# Frozen-feature linear-probe diagnostic for the La-Proteina trunk.
# Reuses the p1zwvmib training run's already-staged h100 /netscratch
# artifacts (venv + preprocessed SwissProt .pt store + La-Proteina ckpts);
# stages/preprocesses nothing. MUST run on the h100 partition — the data is
# only visible from the h100 /netscratch store, not the gpu store / login01.

set -euo pipefail

echo "==========================================="
echo "QG frozen-feature probe started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "==========================================="

PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"
DATA_ROOT="/netscratch/schekmenev"

# --- Stage the venv fresh from the tarball (the pre-existing
# $DATA_ROOT/.venv was a corrupt/partial extraction — torch and hydra
# module trees missing despite dist-info present). Extract idempotently
# into a probe-specific dir so we never mix with the broken one. ---
ENV_TARBALL="$PROJECT_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/ffprobe_venv/.venv"
if [ ! -x "$ENV_LOCAL/bin/python" ]; then
    echo "Extracting venv from $ENV_TARBALL ..."
    mkdir -p "$DATA_ROOT/ffprobe_venv"
    if ! tar xzf "$ENV_TARBALL" -C "$DATA_ROOT/ffprobe_venv"; then
        echo "ERROR: failed to extract $ENV_TARBALL" >&2
        exit 1
    fi
fi
if ! "$ENV_LOCAL/bin/python" -c "import hydra, torch" 2>/dev/null; then
    echo "ERROR: staged venv at $ENV_LOCAL cannot import hydra+torch. Aborting." >&2
    exit 1
fi
echo "venv OK at $ENV_LOCAL"
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# --- Reuse the staged, preprocessed SwissProt store + La-Proteina ckpts ---
SWISSPROT_DIR="$DATA_ROOT/swissprot_v4"
CKPT_DIR="$DATA_ROOT/ckpt/quality_graft"
for f in "$SWISSPROT_DIR/processed/plddt_status.csv" \
         "$CKPT_DIR/LD1_ucond_notri_512.ckpt" \
         "$CKPT_DIR/AE1_ucond_512.ckpt"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: expected staged artifact missing: $f" >&2
        exit 1
    fi
done

N_PT=$(find "$SWISSPROT_DIR/processed" -name "*.pt" | wc -l)
echo "Staged SwissProt .pt files: $N_PT"
echo "Data dir : $SWISSPROT_DIR"
echo "Ckpts    : $CKPT_DIR/{LD1_ucond_notri_512,AE1_ucond_512}.ckpt"

mkdir -p "$PROJECT_ROOT/logs"

# --- Run the probe ---
# min/max length pinned to the staged CSV (df_swissprot_f1.0_minl30_maxl256).
srun --kill-on-bad-exit "$ENV_LOCAL/bin/python" \
    "$PROJECT_ROOT/scripts/run_frozen_feature_probe.py" \
    data=swissprot \
    data.local_only=true \
    data.data_dir="$SWISSPROT_DIR" \
    data.source_dir=/mnt/labs/shared/databases/swissprot_pdb_v4/files \
    data.metadata_tsv="$PROJECT_ROOT/data/metadata/swissprot/uniprot_metadata.tsv" \
    training.min_length=30 \
    training.max_length=256 \
    training.num_workers=4 \
    model.la_proteina_wrapper.proteina_ckpt_path="$CKPT_DIR/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$CKPT_DIR/AE1_ucond_512.ckpt" \
    +probe.max_proteins_train_fit=300 \
    +probe.max_proteins_train_eval=200 \
    +probe.max_proteins_val_eval=200 \
    +probe.fit_steps=3000 \
    +probe.log_every=100 \
    +probe.seed=0 \
    +probe.device=auto \
    +probe.t_value=0.99 \
    +probe.deterministic_encode=true

echo "==========================================="
echo "QG frozen-feature probe finished: $(date)"
echo "==========================================="
