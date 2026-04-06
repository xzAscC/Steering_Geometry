#!/usr/bin/env bash
# =============================================================================
# run_all_models_mmlu_topk.sh - TDNV MMLU across all paper models (top-K)
# =============================================================================
# Same as run_all_models_mmlu.sh but uses --top-k to select only the most
# discriminative tokens per class, reducing noise and compute cost.
#
# Models:
#   Qwen3:   1.7B, 4B
#   Qwen3.5: 9B, 27B
#   Gemma-2: 2B, 9B
#   OLMo-3:  7B, 32B
#
# Usage:
#   ./scripts/tdnv/run_all_models_mmlu_topk.sh                  # All models, top-100
#   ./scripts/tdnv/run_all_models_mmlu_topk.sh -k 50            # Top-50 tokens per class
#   ./scripts/tdnv/run_all_models_mmlu_topk.sh -q 200           # 200 questions per model
#   ./scripts/tdnv/run_all_models_mmlu_topk.sh --dry-run        # Dry run
#
# Output:
#   JSON:  data/tdnv_topk/mmlu_{model}.json
#   Plot:  plot/tdnv_topk/mmlu_{model}.pdf
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
NUM_QUESTIONS=100
MMLU_SEED=42
TOP_K=100
CATEGORIES=""
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv_topk"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv_topk"
DRY_RUN=false

# --- Paper models ---
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
        -q|--questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --seed)
            MMLU_SEED="$2"
            shift 2
            ;;
        --categories)
            CATEGORIES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Run TDNV MMLU analysis across all paper models with top-K token selection."
            echo ""
            echo "Options:"
            echo "  -k, --top-k N       Number of discriminative tokens per class (default: 100)"
            echo "  -q, --questions N   Number of MMLU questions (default: 100)"
            echo "  --seed N            Random seed for question sampling (default: 42)"
            echo "  --categories LIST   Comma-separated category names (default: all)"
            echo "  --dry-run           Dry run (no model loading)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Models: Qwen3 (1.7B, 4B), Qwen3.5 (9B, 27B), Gemma-2 (2B, 9B), OLMo-3 (7B, 32B)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$PLOT_DIR"

total=${#MODELS[@]}
current=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV MMLU All-Models Analysis (Top-K)${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Models:     ${GREEN}${#MODELS[@]} models${NC}"
echo -e "Questions:  ${GREEN}$NUM_QUESTIONS${NC}"
echo -e "Top-K:      ${GREEN}$TOP_K tokens per class${NC}"
echo -e "Seed:       ${GREEN}$MMLU_SEED${NC}"
if [[ -n "$CATEGORIES" ]]; then
    echo -e "Categories: ${GREEN}$CATEGORIES${NC}"
else
    echo -e "Categories: ${GREEN}all${NC}"
fi
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Plots:      ${YELLOW}$PLOT_DIR${NC}"
echo -e "Dry run:    ${YELLOW}$DRY_RUN${NC}"
echo -e "Total:      ${GREEN}$total model(s)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Build args array
TDNV_ARGS=(
    --mode mmlu
    --top-k "$TOP_K"
    --num-questions "$NUM_QUESTIONS"
    --mmlu-seed "$MMLU_SEED"
    --output "$OUTPUT_DIR"
    --plot-dir "$PLOT_DIR"
)

if [[ -n "$CATEGORIES" ]]; then
    TDNV_ARGS+=(--categories "$CATEGORIES")
fi

if [[ "$DRY_RUN" == true ]]; then
    TDNV_ARGS+=(--dry-run)
fi

failed=0

for model in "${MODELS[@]}"; do
    ((++current))
    echo -e "${GREEN}[$current/$total] Analyzing: $model (MMLU, top-$TOP_K)${NC}"
    echo "----------------------------------------"

    if uv run python -m "steering_geometry.tdnv" \
        --model "$model" \
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

echo -e "${BLUE}============================================${NC}"
if [[ $failed -eq 0 ]]; then
    echo -e "${GREEN}✓ All $total MMLU top-K analyses complete!${NC}"
else
    echo -e "${YELLOW}⚠ $failed / $total analyses failed${NC}"
fi
echo -e "${BLUE}============================================${NC}"
echo -e "  JSON results: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  Plots:        ${YELLOW}$PLOT_DIR${NC}"
echo -e "${BLUE}============================================${NC}"

exit $failed
