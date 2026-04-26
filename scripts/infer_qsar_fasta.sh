#!/usr/bin/env bash
set -euo pipefail

# Inference script for QSAR-guided LeadGFlowNet model with FASTA protein sequence
#
# Usage:
#   bash scripts/infer_qsar_fasta.sh [NUM_SAMPLES] [EXTRA_ARGS...]
#
# This script runs inference with:
#   - Trained QSAR-guided model: checkpoints/leadgflownet_online_tb_qsar.pt
#   - QSAR checkpoint for scoring: checkpoints/qsar.pt
#   - FASTA protein sequence for QSAR prediction
#   - QSAR-guided search ordering (--use-qsar)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default parameters
NUM_SAMPLES="${1:-300}"
EXTRA_ARGS=("${@:2}")

# Model checkpoints
MODEL_CKPT="checkpoints/leadgflownet_online_tb_qsar.pt"
QSAR_CKPT="checkpoints/qsar.pt"

# FASTA protein sequence
PROTEIN_SEQ="SDKKSLMPLVGIPGEIKNRLNILDFVKNDKFFTLYVRALQVLQARDQSDYSSFFQLGGIHGLPYTEWAKAQPQLHLYKANYCTHGTVLFPTWHRAYESTWEQTLWEAAGTVAQRFTTSDQAEWIQAAKDLRQPFWDWGYWPNDPDFIGLPDQVIRDKQVEITDYNGTKIEVENPILHYKFHPIEPTFEGDFAQWQTTMRYPDVQKQENIEGMIAGIKAAAPGFREWTFNMLTKNYTWELFSNHGAVVGAHANSLEMVHNTVHFLIGRDPTLDPLVPGHMGSVPHAAFDPIFWMHHCNVDRLLALWQTMNYDVYVSEGMNREATMGLIPGQVLTEDSPLEPFYTKNQDPWQSDDLEDWETLGFSYPDFDPVKGKSKEEKSVYINDWVHKHYG"

# Check checkpoints
if [[ ! -f "$MODEL_CKPT" ]]; then
    echo "Error: Model checkpoint not found: $MODEL_CKPT" >&2
    exit 1
fi

if [[ ! -f "$QSAR_CKPT" ]]; then
    echo "Error: QSAR checkpoint not found: $QSAR_CKPT" >&2
    exit 1
fi

# Output paths
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_JSON="runs/leads_qsar_${TIMESTAMP}.json"
OUTPUT_RANKED_CSV="runs/leads_qsar_ranked_${TIMESTAMP}.csv"
OUTPUT_RANKED_JSON="runs/leads_qsar_ranked_${TIMESTAMP}.json"
PROGRESS_FILE="runs/infer_progress_${TIMESTAMP}.txt"

echo "Starting QSAR-guided inference..."
echo "  - Model checkpoint: $MODEL_CKPT"
echo "  - QSAR checkpoint: $QSAR_CKPT"
echo "  - Protein sequence: ${PROTEIN_SEQ:0:50}..."
echo "  - Number of samples: $NUM_SAMPLES"
echo "  - Output JSON: $OUTPUT_JSON"
echo "  - Ranked CSV: $OUTPUT_RANKED_CSV"
echo ""

/home/jb/miniconda3/envs/phar/bin/python -u leadgflownet_infer.py \
    --input data/reaction_paths_all_routes.csv \
    --forward data/forward_trajectories.csv \
    --checkpoint "$MODEL_CKPT" \
    --protein "$PROTEIN_SEQ" \
    --protein-encoder esm2 \
    --esm2-model lib/models--facebook--esm2_t30_150M_UR50D \
    --num-samples "$NUM_SAMPLES" \
    --max-depth 8 \
    --branch-block-topk 2 \
    --branch-rxn-topk 1 \
    --temperature 1.0 \
    --use-qsar \
    --qsar-checkpoint "$QSAR_CKPT" \
    --qsar-mix 0.7 \
    --output-json "$OUTPUT_JSON" \
    --export-ranked \
    --output-ranked-csv "$OUTPUT_RANKED_CSV" \
    --output-ranked-json "$OUTPUT_RANKED_JSON" \
    --progress-file "$PROGRESS_FILE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[OK] Inference finished."
echo "  - Routes JSON: $OUTPUT_JSON"
echo "  - Ranked CSV: $OUTPUT_RANKED_CSV"
echo "  - Ranked JSON: $OUTPUT_RANKED_JSON"
echo "  - Progress file: $PROGRESS_FILE"

