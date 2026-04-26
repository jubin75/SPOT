#!/usr/bin/env bash
set -euo pipefail

# Offline GFlowNet Trajectory Balance training wrapper
#
# Usage:
#   bash scripts/offline_tb_train.sh [EPOCHS] [BATCH_SIZE]
#
# Uses full datasets by default (max_trajectories=0).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

EPOCHS="${1:-15}"
BATCH="${2:-128}"
# Forward any extra CLI args to the trainer
EXTRA_ARGS=("${@:3}")

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

torchrun --standalone --nproc_per_node=2 -m LeadGFlowNet.offline_tb_train \
  --distributed ddp \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --pactivity "$PACT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr 3e-4 \
  --max-trajectories 0 \
  --protein-encoder esm2 \
  --esm2-model lib/models--facebook--esm2_t30_150M_UR50D \
  --use-checkpoint \
  --checkpoint checkpoints/synth_policy_net.pt \
  --save checkpoints/leadgflownet_offline_tb.pt
  "${EXTRA_ARGS[@]}"

echo "[OK] Offline TB checkpoint saved to checkpoints/leadgflownet_offline_tb.pt"



