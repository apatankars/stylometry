#!/usr/bin/env bash
# V2 ablation suite — all 6 experiments with locked shared hyperparameters.
#
# Shared across all runs:
#   loss:         supcon, temperature=0.07
#   data:         100 train senders, K=8 emails/sender, 10% val/test splits
#   training:     20 epochs, batch_size=64, cosine LR schedule, grad_clip=1.0
#   encoder:      projection_dim=128, max_length=512
#
# Variable by design (appropriate per architecture):
#   frozen runs:  lr=1e-3, warmup=50 steps  (only ~98K projection params)
#   LoRA runs:    lr=2e-4, warmup=200 steps (adapter + projection, ~1.2M params)
#
# LUAR runs use episode_k=4: 8 emails/sender → 2 episodes × 4 emails each.
# The val loader uses PKSampler for LUAR to preserve P×K batch structure.
#
# Estimated wall-clock (A100 80GB): ~15 min/run for frozen, ~40 min for LoRA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

EXPERIMENTS=(
    configs/experiments/v2_roberta_frozen_proj.yaml
    configs/experiments/v2_mpnet_frozen_proj.yaml
    configs/experiments/v2_roberta_lora_proj.yaml
    configs/experiments/v2_mpnet_lora_proj.yaml
    configs/experiments/v2_luar_frozen_proj.yaml
    configs/experiments/v2_luar_lora.yaml
)

for cfg in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    python scripts/train.py --config "$cfg"
done

echo ""
echo "All V2 ablation experiments complete."
