#!/usr/bin/env bash
# V6 ablation suite — 2x2 over (backbone trainable) x (synthetic hard negatives),
# LUAR-only.
#
#                  │ no-synthetic        with-synthetic
#   ───────────────┼──────────────────────────────────────
#   LUAR frozen    │ v6_luar_frozen      v6_luar_frozen_syn
#   LUAR LoRA      │ v6_luar_lora        v6_luar_lora_syn
#
# Two questions in one suite:
#   Q1: Does synthetic hard-negative augmentation help LUAR?  (within-row pairs)
#   Q2: Does fine-tuning (LoRA) on top of LUAR-MUD beat using LUAR frozen as a
#       pure feature extractor?  (across-row pairs)
#
# Comparability — locked across all four runs:
#   loss:                   supcon, temperature=0.07
#   data:                   100 train senders, K=8 emails/sender, 10/10 val/test
#   batch_size:             64
#   scheduler:              cosine, warmup=200
#   grad_clip:              1.0
#   mixed_precision:        true
#   epochs:                 100
#   early_stopping_patience:8 (on val/loss, min_delta=1e-4)
#
# Variable by design:
#   frozen rows:  lr=1e-3   (only projection head trains)
#   LoRA rows:    lr=2e-4   (LoRA adapters + projection head train)
#
# Each run will stop early once val/loss hasn't improved for 8 consecutive
# epochs, so wall-clock varies. With patience=8 on a typical SupCon curve we
# expect 25-45 epochs in practice; the 100-epoch budget is a hard cap.
#
# Usage:
#   bash scripts/run_experiments_v6.sh                       # all four runs sequentially
#   bash scripts/run_experiments_v6.sh frozen                # only the frozen row
#   bash scripts/run_experiments_v6.sh lora                  # only the LoRA row
#   bash scripts/run_experiments_v6.sh syn                   # only the synthetic column
#   bash scripts/run_experiments_v6.sh baseline              # only the no-synthetic column
#   bash scripts/run_experiments_v6.sh v6_luar_lora_syn      # a single named experiment
#   bash scripts/run_experiments_v6.sh v6_luar_lora_syn.yaml # also accepted (with extension)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

FROZEN_BASELINE="configs/experiments/v6_luar_frozen.yaml"
FROZEN_SYN="configs/experiments/v6_luar_frozen_syn.yaml"
LORA_BASELINE="configs/experiments/v6_luar_lora.yaml"
LORA_SYN="configs/experiments/v6_luar_lora_syn.yaml"

case "${1:-all}" in
    frozen)
        EXPERIMENTS=("$FROZEN_BASELINE" "$FROZEN_SYN")
        ;;
    lora)
        EXPERIMENTS=("$LORA_BASELINE" "$LORA_SYN")
        ;;
    syn)
        EXPERIMENTS=("$FROZEN_SYN" "$LORA_SYN")
        ;;
    baseline|nosyn)
        EXPERIMENTS=("$FROZEN_BASELINE" "$LORA_BASELINE")
        ;;
    all|"")
        EXPERIMENTS=(
            "$FROZEN_BASELINE"
            "$FROZEN_SYN"
            "$LORA_BASELINE"
            "$LORA_SYN"
        )
        ;;
    *)
        candidate="$1"
        candidate="${candidate%.yaml}"
        cfg_path="configs/experiments/${candidate}.yaml"
        if [[ -f "$cfg_path" ]]; then
            EXPERIMENTS=("$cfg_path")
        else
            echo "Unknown subset or missing config: $1" >&2
            echo "Usage: $0 [all|frozen|lora|syn|baseline|<exp_name>]" >&2
            echo "Looked for: $cfg_path" >&2
            exit 2
        fi
        ;;
esac

# Pre-flight: synthetic dataset must exist for any *_syn run.
NEEDS_SYN=0
for cfg in "${EXPERIMENTS[@]}"; do
    case "$cfg" in *_syn.yaml) NEEDS_SYN=1 ;; esac
done
if [[ "$NEEDS_SYN" -eq 1 && ! -d "data/synthetic/enron_synthetic" ]]; then
    echo "ERROR: data/synthetic/enron_synthetic is missing." >&2
    echo "Generate it first with:" >&2
    echo "  python scripts/generate_synthetic_emails.py --config configs/base.yaml \\" >&2
    echo "      --output data/synthetic/enron_synthetic --load-in-4bit" >&2
    exit 1
fi

# Pretty-printing helpers -------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

model_name_for() {
    local cfg="$1"
    local name
    name=$(awk '
        /^encoder:/ { in_enc=1; next }
        in_enc && /^[^[:space:]]/ { in_enc=0 }
        in_enc && /model_name_or_path:/ {
            sub(/.*model_name_or_path:[[:space:]]*/, "")
            gsub(/["'\''[:space:]]/, "")
            print; exit
        }
    ' "$cfg")
    echo "${name:-unknown-model}"
}

fmt_duration() {
    local s=$1
    printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

BAR="────────────────────────────────────────────────────────────────────"
TOTAL=${#EXPERIMENTS[@]}

printf '\n%s%s V6 ablation suite %s— %d experiment(s)%s\n' \
    "$BOLD" "$CYAN" "$RESET$BOLD" "$TOTAL" "$RESET"
printf '%s%s%s\n' "$DIM" "$BAR" "$RESET"
i=0
for cfg in "${EXPERIMENTS[@]}"; do
    i=$((i + 1))
    printf '  %s[%d/%d]%s %-50s %s%s%s\n' \
        "$DIM" "$i" "$TOTAL" "$RESET" \
        "$(basename "$cfg")" \
        "$DIM" "$(model_name_for "$cfg")" "$RESET"
done
printf '%s%s%s\n\n' "$DIM" "$BAR" "$RESET"

SUITE_START=$(date +%s)
i=0
FAILED=()
for cfg in "${EXPERIMENTS[@]}"; do
    i=$((i + 1))
    model=$(model_name_for "$cfg")
    exp_name=$(basename "${cfg%.yaml}")

    printf '\n%s%s%s\n' "$CYAN" "$BAR" "$RESET"
    printf '%s▶ [%d/%d] %s%s\n'   "$BOLD$CYAN" "$i" "$TOTAL" "$exp_name" "$RESET"
    printf '   %sconfig:%s %s\n'  "$DIM" "$RESET" "$cfg"
    printf '   %smodel:%s  %s%s%s\n' "$DIM" "$RESET" "$BOLD" "$model" "$RESET"
    printf '   %sstart:%s  %s\n'  "$DIM" "$RESET" "$(date -Iseconds)"
    printf '%s%s%s\n' "$CYAN" "$BAR" "$RESET"

    run_start=$(date +%s)
    if python scripts/train.py --config "$cfg"; then
        elapsed=$(( $(date +%s) - run_start ))
        printf '%s✓ %s finished in %s%s\n' \
            "$GREEN" "$exp_name" "$(fmt_duration "$elapsed")" "$RESET"
    else
        rc=$?
        elapsed=$(( $(date +%s) - run_start ))
        printf '%s✗ %s failed (exit %d) after %s%s\n' \
            "$RED" "$exp_name" "$rc" "$(fmt_duration "$elapsed")" "$RESET"
        FAILED+=("$exp_name")
    fi
done

SUITE_ELAPSED=$(( $(date +%s) - SUITE_START ))
printf '\n%s%s%s\n' "$DIM" "$BAR" "$RESET"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    printf '%s✓ All %d experiments finished in %s%s\n' \
        "$GREEN$BOLD" "$TOTAL" "$(fmt_duration "$SUITE_ELAPSED")" "$RESET"
else
    printf '%s%d/%d experiments failed%s: %s\n' \
        "$YELLOW$BOLD" "${#FAILED[@]}" "$TOTAL" "$RESET" "${FAILED[*]}"
fi
printf '%s%s%s\n' "$DIM" "$BAR" "$RESET"
echo ""
echo "Compare runs in W&B side-by-side on these axes:"
echo "  auc/genuine_vs_other"
echo "  auc/genuine_vs_synthetic                   <-- the hard-negative number"
echo "  score/synthetic_harder_than_other          <-- positive ⇒ syn is harder than other-sender (good)"
echo "  coverage/at_acc_0.95                       <-- selective coverage (scale-free)"
echo "  embedding/pair_auroc                       <-- sender-disjoint generalization on val batches"
echo "  early_stopped_at_epoch                     <-- which runs hit the patience limit"
echo "  test/auc, test/eer                         <-- final headline numbers"
echo ""
echo "NOTE: threshold_0.95/* is unreachable under the current PrototypicalHead"
echo "score mapping (score = 1 - z/3); use coverage/at_acc_0.95 for high-confidence behaviour."
