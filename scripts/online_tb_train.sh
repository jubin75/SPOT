#!/usr/bin/env bash
set -euo pipefail

# Online QSAR-only fine-tuning wrapper (saves to checkpoints/leadgflownet_online_tb.pt)
#
# Usage:
#   bash scripts/online_tb_train.sh [EPOCHS]
#
# Defaults:
#   - Uses curated pActivity CSV if present, else falls back to raw pActivity
#   - Uses offline TB checkpoint if present, else falls back to SynthPolicyNet pretrain

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

EPOCHS="${1:-5}"
# Forward any extra CLI args to the trainer
EXTRA_ARGS=("${@:2}")

# Stabilize tokenizer/BLAS threading and prefer offline transformers
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

# Prefer curated pActivity if present
PACT="data/protein_ligand_pactivity_curated.csv"
if [[ ! -f "$PACT" ]]; then
  PACT="data/protein_ligand_pactivity.csv"
fi

# Prefer offline TB checkpoint; fallback to SynthPolicyNet pretrain
CKPT="checkpoints/leadgflownet_offline_tb.pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="checkpoints/synth_policy_net.pt"
fi

# Decide whether to use DDP based on EXTRA_ARGS and GPU count
JOINED_ARGS=" ${EXTRA_ARGS[*]} "
USE_DDP=1
if echo "$JOINED_ARGS" | grep -qE ' --distributed(=| )none( |$)'; then
  USE_DDP=0
fi
# Count visible GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpus <<<"$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_gpus[@]}
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NUM_GPUS=0
  fi
fi
if [[ "$NUM_GPUS" -lt 2 ]]; then
  USE_DDP=0
fi

SAVE_PATH="checkpoints/leadgflownet_online_tb.pt"

if [[ "$USE_DDP" -eq 1 ]]; then
  torchrun --standalone --nproc_per_node=2 -m LeadGFlowNet.online_tb_train \
    --distributed ddp \
    --input data/reaction_paths_all_routes.csv \
    --forward data/forward_trajectories.csv \
    --checkpoint "$CKPT" \
    --qsar-checkpoint checkpoints/qsar.pt \
    --pactivity "$PACT" \
    --epochs "$EPOCHS" \
    --episodes-per-epoch 800 \
    --max-steps 8 \
    --lr 3e-4 \
    --save "$SAVE_PATH" \
    --rxn-first \
    "${EXTRA_ARGS[@]}"
else
  python -u -m LeadGFlowNet.online_tb_train \
    --input data/reaction_paths_all_routes.csv \
    --forward data/forward_trajectories.csv \
    --checkpoint "$CKPT" \
    --qsar-checkpoint checkpoints/qsar.pt \
    --pactivity "$PACT" \
    --epochs "$EPOCHS" \
    --episodes-per-epoch 800 \
    --max-steps 8 \
    --lr 3e-4 \
    --save "$SAVE_PATH" \
    --rxn-first \
    "${EXTRA_ARGS[@]}"
fi

echo "[OK] Online TB finished. Saved to $SAVE_PATH"


