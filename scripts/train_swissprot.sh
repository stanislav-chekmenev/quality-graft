#!/bin/bash
set -euo pipefail

echo "==========================================="
echo "SwissProt training started: $(date)"
echo "Running on: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# --- Environment ---
PROJECT_ROOT="/mnt/labs/home/schekmenev/projects/quality-graft"

NUM_DEVICES=1
DEVICE_TYPE="$DEVICE_TYPE"

if [ "$DEVICE_TYPE" == "rtx6000" ]; then
    DATA_ROOT="/scratch/schekmenev"
elif [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/schekmenev"
else
    echo "ERROR: Unknown DEVICE_TYPE=$DEVICE_TYPE. Set to rtx6000 or h100nvl."
    exit 1
fi

# Activate project venv
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_ROOT/logs"

# --- Copy SwissProt data to scratch ---
SWISSPROT_DIR="$DATA_ROOT/swissprot_v4"
METADATA_TSV="$PROJECT_ROOT/data/metadata/swissprot/uniprot_metadata.tsv"

# SwissProt filtering parameters
MIN_LENGTH=30
MAX_LENGTH=31

echo "Copying filtered SwissProt PDBs to $SWISSPROT_DIR/raw ..."
python "$PROJECT_ROOT/scripts/copy_swissprot.py" \
    --source-dir /mnt/labs/shared/databases/swissprot_pdb_v4/files \
    --dest-dir "$SWISSPROT_DIR/raw" \
    --metadata-tsv "$METADATA_TSV" \
    --min-length $MIN_LENGTH \
    --max-length $MAX_LENGTH

N_PDB=$(find "$SWISSPROT_DIR/raw" -name "*.pdb" | wc -l)
echo "SwissProt raw PDBs on scratch: $N_PDB"
if [ "$N_PDB" -eq 0 ]; then
    echo "ERROR: No PDB files copied. Aborting."
    exit 1
fi

# --- Stage La-Proteina model weights to scratch ---
WEIGHT_DIR="$DATA_ROOT/ckpt"
mkdir -p "$WEIGHT_DIR"
echo "Staging La-Proteina model weights to $WEIGHT_DIR..."
# Copy the weights only for La-Proteina, but skip the boltz1_conf.ckpt
for ckpt in "$PROJECT_ROOT/ckpt/"*.ckpt; do
    if [[ $(basename "$ckpt") != "boltz1_conf.ckpt" ]]; then
        cp -v "$ckpt" "$WEIGHT_DIR/"
    fi
done
echo "La-Proteina weight staging complete."

# Training checkpoint directory (scratch, fast I/O)
TRAIN_CKPT_DIR="$DATA_ROOT/ckpt/runs"
mkdir -p "$TRAIN_CKPT_DIR"

echo ""
echo "Configuration:"
echo "  Data dir:    $SWISSPROT_DIR"
echo "  Weights:     $WEIGHT_DIR"
echo "  Checkpoints: $TRAIN_CKPT_DIR"
echo "  GPUs:        ${NUM_DEVICES}x $DEVICE_TYPE"
echo ""

# --- Run training ---
echo "Starting training (student distillation mode)..."
python "$PROJECT_ROOT/scripts/train.py" \
    mode=train \
    data=swissprot \
    data.data_dir="$SWISSPROT_DIR" \
    data.source_dir=/mnt/labs/shared/databases/swissprot_pdb_v4/files \
    data.metadata_tsv="$METADATA_TSV" \
    data.batch_size=6 \
    data.num_workers=6 \
    training.max_length=$MAX_LENGTH \
    training.min_length=$MIN_LENGTH \
    training.accelerator=gpu \
    training.devices=$NUM_DEVICES \
    training.strategy=auto \
    training.precision=bf16-mixed \
    training.max_epochs=100 \
    training.check_val_every_n_epoch=10 \
    training.checkpoint_dir="$TRAIN_CKPT_DIR" \
    model.adaptor.n_attn_layers=0 \
    model.la_proteina_wrapper.proteina_ckpt_path="$WEIGHT_DIR/LD1_ucond_notri_512.ckpt" \
    model.la_proteina_wrapper.autoencoder_ckpt_path="$WEIGHT_DIR/AE1_ucond_512.ckpt" \
    model/confidence_head=student \


# --- Copy checkpoints back ---
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
