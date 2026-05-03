#!/usr/bin/env bash
# V3 LoRA fix — corrected hyperparameters for RoBERTa and MPNet.
#
# V2 post-mortem: both LoRA runs (RoBERTa 0.526, MPNet 0.485 AUROC) fell
# below or near random.  Root cause: lr=2e-4 + r=16 + 20 epochs too
# aggressive.  V3 fixes: lr=5e-5, warmup=500, r=8, epochs=40.
#
# LUAR LoRA (AUROC 0.956 in V2) is excluded — it already converged well.
# These two runs are specifically ablating the failed V2 LoRA settings.
#
# Estimated wall-clock (A100 80GB): ~60–90 min/run (2x epochs vs V2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

EXPERIMENTS=(
    configs/experiments/v3_roberta_lora.yaml
    configs/experiments/v3_mpnet_lora.yaml
)

for cfg in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    python scripts/train.py --config "$cfg"
done

echo ""
echo "V3 LoRA fix experiments complete."
echo ""
echo "Evaluate on test set:"
echo "  python scripts/eval_checkpoint.py --list"
echo "  python scripts/eval_checkpoint.py --run runs/v3_roberta_lora/<timestamp>"
echo "  python scripts/eval_checkpoint.py --run runs/v3_mpnet_lora/<timestamp>"
