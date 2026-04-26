#!/usr/bin/env bash
set -euo pipefail

# Online TB training with QSAR model guidance using FASTA protein sequence
#
# Usage:
#   bash scripts/online_tb_train_qsar_fasta.sh <FASTA_FILE_OR_SEQUENCE> [EPOCHS] [EXTRA_ARGS...]
#
# Examples:
#   # From FASTA file:
#   bash scripts/online_tb_train_qsar_fasta.sh data/target_protein.fasta 5
#
#   # Direct sequence string:
#   bash scripts/online_tb_train_qsar_fasta.sh "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL" 5
#
# This script runs Online TB training with:
#   - QSAR model for reward prediction (--use-qsar-reward)
#   - FASTA sequence as protein input (--protein-fasta)
#   - QSAR checkpoint for ligand activity prediction

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <FASTA_FILE_OR_SEQUENCE> [EPOCHS] [EXTRA_ARGS...]"
    echo ""
    echo "Examples:"
    echo "  # From FASTA file:"
    echo "  $0 data/target_protein.fasta 5"
    echo ""
    echo "  # Direct sequence string:"
    echo "  $0 \"MKTAYIAKQRQISFVKSHFSRQ...\" 5"
    echo ""
    exit 1
fi

PROTEIN_INPUT="$1"
EPOCHS="${2:-5}"
EXTRA_ARGS=("${@:3}")

# Check QSAR checkpoint
QSAR_CKPT="checkpoints/qsar.pt"
if [[ ! -f "$QSAR_CKPT" ]]; then
    echo "Error: QSAR checkpoint not found: $QSAR_CKPT" >&2
    echo "Please train QSAR model first:" >&2
    echo "  python -m LeadGFlowNet.qsar train_qsar data/protein_ligand_pactivity.csv" >&2
    exit 1
fi

# Prefer offline TB checkpoint; fallback to SynthPolicyNet pretrain
CKPT="checkpoints/leadgflownet_offline_tb.pt"
if [[ ! -f "$CKPT" ]]; then
    CKPT="checkpoints/synth_policy_net.pt"
fi

# Stabilize tokenizer/BLAS threading
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

SAVE_PATH="checkpoints/leadgflownet_online_tb_qsar.pt"

echo "Starting Online TB training with QSAR guidance..."
echo "  - Protein input: $PROTEIN_INPUT"
echo "  - QSAR checkpoint: $QSAR_CKPT"
echo "  - Base checkpoint: $CKPT"
echo "  - Epochs: $EPOCHS"
echo ""

/home/jb/miniconda3/envs/phar/bin/python -u -m LeadGFlowNet.online_tb_train \
    --input data/reaction_paths_all_routes.csv \
    --forward data/forward_trajectories.csv \
    --checkpoint "$CKPT" \
    --qsar-checkpoint "$QSAR_CKPT" \
    --protein-fasta "$PROTEIN_INPUT" \
    --use-qsar-reward \
    --epochs "$EPOCHS" \
    --episodes-per-epoch 800 \
    --max-steps 8 \
    --lr 3e-4 \
    --save "$SAVE_PATH" \
    --rxn-first \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[OK] Online TB training finished. Saved to $SAVE_PATH"

