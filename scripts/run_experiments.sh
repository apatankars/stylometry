#!/usr/bin/env bash
# Run all five experiments sequentially.
# Each writes to runs/<config-stem>/<timestamp>/ and logs to W&B under its config name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

EXPERIMENTS=(
    configs/experiments/roberta_frozen.yaml
    configs/experiments/mpnet_frozen.yaml
    configs/experiments/roberta_full_supcon.yaml
    configs/experiments/roberta_lora_supcon.yaml
    configs/experiments/roberta_lora_triplet.yaml
)

for cfg in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    python scripts/train.py --config "$cfg"
done

echo ""
echo "All experiments complete."
