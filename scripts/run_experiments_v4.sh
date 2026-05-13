#!/usr/bin/env bash
# V4 ablation suite — 2x2 over (architecture) x (synthetic hard negatives).
#
# Tests two questions in one suite:
#   Q1: Does synthetic hard-negative augmentation improve stylometric
#       discrimination?  (within-architecture pairs)
#   Q2: Is the effect architecture-dependent?  (across-architecture)
#
#                 │ no-synthetic      with-synthetic
#   ──────────────┼──────────────────────────────────────────────
#   LUAR LoRA     │ v2_luar_lora      v4_luar_lora_syn
#   RoBERTa LoRA  │ v3_roberta_lora   v4_roberta_lora_syn
#
# Why these four:
#   - LUAR LoRA was the V2 winner (AUROC 0.956). It's the strongest baseline,
#     so we want to know if synthetic still helps a model that already works.
#   - RoBERTa LoRA needed the V3 fix (lr=5e-5, r=8, warmup=500, 40 ep) to be
#     competitive.  Pairing v3 with synthetic isolates the augmentation effect
#     on a fixed, working RoBERTa baseline (NOT the broken v2 settings).
#   - The two no-synthetic configs were never re-run with the new metrics
#     (val/centroid/*, threshold-band reports, coverage@accuracy), so this
#     suite re-runs them to give a directly comparable metric set.
#
# Comparability — what's locked across all four runs:
#   loss:           supcon, temperature=0.07
#   data:           100 train senders, K=8 emails/sender, 10/10 val/test
#   batch_size:     64
#   scheduler:      cosine
#   grad_clip:      1.0
#   mixed_precision: true
#   centroid probe: same enrollment/query splits (built deterministically
#                   from train+val with seed=0 in train.py).
#
# Variable by design (architecture-appropriate, per docs/steps.md item 1):
#   LUAR LoRA:     lr=2e-4, r=16, warmup=200, epochs=20  (V2 settings)
#   RoBERTa LoRA:  lr=5e-5, r=8,  warmup=500, epochs=40  (V3 hyperparam fix)
#
# Within-architecture, the synthetic and non-synthetic configs share every
# hyperparameter except `data.augmentation.synthetic_path` and the
# `n_syn_per_batch` slot.  Any AUROC delta within a row is attributable to
# the augmentation alone.
#
# Estimated wall-clock (A100 80GB), sequential:
#   v2_luar_lora              ~40 min
#   v4_luar_lora_syn          ~50 min  (slightly larger train set)
#   v3_roberta_lora           ~75 min  (40 epochs)
#   v4_roberta_lora_syn       ~90 min
#   ─────────────────────────────────────
#   total                     ~4.5 h
#
# Usage:
#   bash scripts/run_experiments_v4.sh             # all four runs sequentially
#   bash scripts/run_experiments_v4.sh luar        # only the LUAR row
#   bash scripts/run_experiments_v4.sh roberta     # only the RoBERTa row
#   bash scripts/run_experiments_v4.sh syn         # only the synthetic column

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LUAR_BASELINE="configs/experiments/v2_luar_lora.yaml"
LUAR_SYN="configs/experiments/v4_luar_lora_syn.yaml"
ROBERTA_BASELINE="configs/experiments/v3_roberta_lora.yaml"
ROBERTA_SYN="configs/experiments/v4_roberta_lora_syn.yaml"

case "${1:-all}" in
    luar)
        EXPERIMENTS=("$LUAR_BASELINE" "$LUAR_SYN")
        ;;
    roberta)
        EXPERIMENTS=("$ROBERTA_BASELINE" "$ROBERTA_SYN")
        ;;
    syn)
        EXPERIMENTS=("$LUAR_SYN" "$ROBERTA_SYN")
        ;;
    baseline|nosyn)
        EXPERIMENTS=("$LUAR_BASELINE" "$ROBERTA_BASELINE")
        ;;
    all|"")
        EXPERIMENTS=(
            "$LUAR_BASELINE"
            "$LUAR_SYN"
            "$ROBERTA_BASELINE"
            "$ROBERTA_SYN"
        )
        ;;
    *)
        echo "Unknown subset: $1" >&2
        echo "Usage: $0 [all|luar|roberta|syn|baseline]" >&2
        exit 2
        ;;
esac

# Pre-flight: synthetic dataset must exist for the syn runs.
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

# Pull the model display name from the YAML's encoder.model_name_or_path so
# the banner reflects what's actually being trained, not just the config stem.
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

printf '\n%s%s V4 ablation suite %s— %d experiment(s)%s\n' \
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
echo "  val/centroid/auroc_genuine_vs_other"
echo "  val/centroid/auroc_genuine_vs_synthetic   <-- the hard-negative number"
echo "  val/centroid/synthetic_harder              <-- positive ⇒ syn is harder than other-sender (good)"
echo "  val/centroid/precision@0.95"
echo "  val/centroid/fpr_synthetic@0.95            <-- low ⇒ high-confidence verdicts resist LLM mimicry"
echo "  val/centroid/coverage_at_acc@0.95"
echo "  test/auc, test/eer                         <-- final headline numbers"
echo ""
echo "Then run the deeper post-hoc probe on the best checkpoint:"
echo "  python scripts/score_centroids.py --run runs/<exp>/<ts> \\"
echo "      --include-synthetic data/synthetic/enron_synthetic --wandb"
