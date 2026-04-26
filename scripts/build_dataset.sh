#!/usr/bin/env bash
set -euo pipefail

# Build retrosynthesis routes and forward trajectories
#
# Usage:
#   bash scripts/build_dataset.sh [LIMIT]
#
# Input files:
#   data/protein_ligand_pactivity.csv (expects column 'ligand_smiles')
#   Optional: data/protein_ligand_pactivity_curated.csv (produced by filter step below)
# Outputs:
#   data/reaction_paths_all_routes.csv
#   data/forward_trajectories.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

LIMIT="${1:-20}"

CURATED="data/protein_ligand_pactivity_curated.csv"
SOURCE="data/protein_ligand_pactivity.csv"

if [[ -f "$CURATED" ]]; then
  echo "Detected curated input: $CURATED"
  INPUT_FILE="$CURATED"
else
  INPUT_FILE="$SOURCE"
fi

echo "[1/2] Building retrosynthesis routes with AiZynthFinder ..."
python -m scripts.build_all_routes_dataset \
  --config config.yml \
  --input "$INPUT_FILE" \
  --limit "$LIMIT" \
  --output data/reaction_paths_all_routes.csv

echo "[2/2] Converting to forward trajectories ..."
python -m scripts.forward_trajectories \
  --input data/reaction_paths_all_routes.csv \
  --output data/forward_trajectories.csv \
  --skip-start-steps

echo "[OK] Dataset built at data/forward_trajectories.csv"


