#!/usr/bin/env bash
# =============================================================================
# run_all_models.sh - TDNV across all paper models × all concepts
# =============================================================================
# Runs TDNV analysis on 8 models (4 families, 2 scales each) across 3 concepts.
# Hyperparameters are fixed to match quick_tdnv.sh defaults.
#
# Active models:
#   Qwen3:   1.7B, 4B
#   Qwen3.5: 9B
#   Gemma-2: 2B, 9B
#   OLMo-3:  7B
#
# Commented out (large, need more GPU):
#   Qwen3.5: 27B, OLMo-3: 32B
#
# Concepts: polite, sentiment, refusal
#
# Usage:
#   ./scripts/tdnv/run_all_models.sh                  # All 8 models × 3 concepts
#   ./scripts/tdnv/run_all_models.sh --dry-run        # Dry run
#
# Output:
#   JSON:  data/tdnv/{concept}_{model}.json
#   Plot:  plot/tdnv/{concept}_{model}.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Fixed hyperparameters (same as quick_tdnv.sh) ---
CONCEPTS=("polite" "sentiment" "refusal")
NUM_PAIRS=100
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv"
DRY_RUN=false
LAST_N=10

# --- 8 paper models ---
MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3.5-9B"
    #"Qwen/Qwen3.5-27B"
    "google/gemma-2-2b"
    "google/gemma-2-9b"
    "allenai/Olmo-3-1025-7B"
    #"allenai/Olmo-3-1125-32B"
)

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--dry-run] [-h|--help]"
            echo ""
            echo "Run TDNV analysis on 6 models × 3 concepts."
            echo "Models: Qwen3 (1.7B, 4B), Qwen3.5 (9B), Gemma-2 (2B, 9B), OLMo-3 (7B)"
            echo "Concepts: polite, sentiment, refusal"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$PLOT_DIR"

total=$((${#CONCEPTS[@]} * ${#MODELS[@]}))
current=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV All-Models Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:   ${GREEN}${CONCEPTS[*]}${NC}"
echo -e "Models:     ${GREEN}${#MODELS[@]} models (Qwen3×2, Qwen3.5×1, Gemma-2×2, OLMo-3×1)${NC}"
echo -e "Pairs:      ${GREEN}$NUM_PAIRS${NC}"
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Plots:      ${YELLOW}$PLOT_DIR${NC}"
echo -e "Dry run:    ${YELLOW}$DRY_RUN${NC}"
echo -e "Total:      ${GREEN}$total analysis(es)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Build optional args array for safe interpolation
TDNV_ARGS=()
if [[ "$DRY_RUN" == true ]]; then
    TDNV_ARGS+=(--dry-run)
fi
if [[ -n "${LAST_N:-}" ]]; then
    TDNV_ARGS+=(--last-n "$LAST_N")
fi

failed=0

for model in "${MODELS[@]}"; do
    for concept in "${CONCEPTS[@]}"; do
        ((++current))
        echo -e "${GREEN}[$current/$total] Analyzing: $concept × $model${NC}"
        echo "----------------------------------------"

        if uv run python -m "steering_geometry.tdnv" \
            --concept "$concept" \
            --model "$model" \
            --num-pairs "$NUM_PAIRS" \
            --output "$OUTPUT_DIR" \
            --plot-dir "$PLOT_DIR" \
            "${TDNV_ARGS[@]}" 2>&1 | while read -r line; do
                echo "  $line"
            done; then
            echo -e "  ${GREEN}✓ Done${NC}"
        else
            echo -e "  ${RED}✗ FAILED${NC}"
            ((++failed))
        fi

        echo ""
    done
done

echo -e "${BLUE}============================================${NC}"
if [[ $failed -eq 0 ]]; then
    echo -e "${GREEN}✓ All $total analyses complete!${NC}"
else
    echo -e "${YELLOW}⚠ $failed / $total analyses failed${NC}"
fi
echo -e "${BLUE}============================================${NC}"
echo -e "  JSON results: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  Plots:        ${YELLOW}$PLOT_DIR${NC}"
echo -e "${BLUE}============================================${NC}"

exit $failed
