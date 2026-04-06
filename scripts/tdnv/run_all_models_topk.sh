#!/usr/bin/env bash
# =============================================================================
# run_all_models_topk.sh - TDNV across all paper models × all concepts (top-K)
# =============================================================================
# Same as run_all_models.sh but uses --top-k to select only the most
# discriminative tokens per class, reducing noise and compute cost.
#
# Active models:
#   Qwen3:   1.7B, 4B
#   Gemma-2: 2B
#   OLMo-3:  7B
#
# Commented out (large, need more GPU):
#   Qwen3.5: 9B, 27B, Gemma-2: 9B, OLMo-3: 32B
#
# Concepts: polite, sentiment, refusal
#
# Usage:
#   ./scripts/tdnv/run_all_models_topk.sh                  # All 8 models × 3 concepts, top-10
#   ./scripts/tdnv/run_all_models_topk.sh -k 20            # Top-20 tokens per class
#   ./scripts/tdnv/run_all_models_topk.sh --dry-run        # Dry run
#
# Output:
#   JSON:  data/tdnv_topk/{concept}_{model}.json
#   Plot:  plot/tdnv_topk/{concept}_{model}.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Fixed hyperparameters ---
CONCEPTS=("polite" "sentiment" "refusal")
NUM_PAIRS=100
TOP_K=100
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv_topk"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv_topk"
DRY_RUN=false

# --- 8 paper models ---
MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    #"Qwen/Qwen3.5-9B"
    #"Qwen/Qwen3.5-27B"
    "google/gemma-2-2b"
    #"google/gemma-2-9b"
    "allenai/Olmo-3-1025-7B"
    #"allenai/Olmo-3-1125-32B"
)

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [-k N] [--dry-run] [-h|--help]"
            echo ""
            echo "Run TDNV analysis on 8 models × 3 concepts with top-K token selection."
            echo ""
            echo "Options:"
            echo "  -k, --top-k N   Number of discriminative tokens per class (default: 100)"
            echo "  --dry-run       Dry run (no model loading)"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Models: Qwen3 (1.7B, 4B), Gemma-2 (2B), OLMo-3 (7B)"
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
echo -e "${BLUE}TDNV All-Models Analysis (Top-K)${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:   ${GREEN}${CONCEPTS[*]}${NC}"
echo -e "Models:     ${GREEN}${#MODELS[@]} models (Qwen3×2, Gemma-2×1, OLMo-3×1)${NC}"
echo -e "Pairs:      ${GREEN}$NUM_PAIRS${NC}"
echo -e "Top-K:      ${GREEN}$TOP_K tokens per class${NC}"
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Plots:      ${YELLOW}$PLOT_DIR${NC}"
echo -e "Dry run:    ${YELLOW}$DRY_RUN${NC}"
echo -e "Total:      ${GREEN}$total analysis(es)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Build optional args array for safe interpolation
TDNV_ARGS=(--top-k "$TOP_K")
if [[ "$DRY_RUN" == true ]]; then
    TDNV_ARGS+=(--dry-run)
fi

failed=0

for model in "${MODELS[@]}"; do
    for concept in "${CONCEPTS[@]}"; do
        ((++current))
        echo -e "${GREEN}[$current/$total] Analyzing: $concept × $model (top-$TOP_K)${NC}"
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
