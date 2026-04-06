#!/usr/bin/env bash
# =============================================================================
# quick_tdnv_mmlu.sh - TDNV analysis across MMLU-Pro categories
# =============================================================================
# Computes separability metrics across MMLU subject categories (math, physics,
# chemistry, etc.) for all model layers.
#
# Usage:
#   ./scripts/tdnv/quick_tdnv_mmlu.sh                                # Default model, 100 questions
#   ./scripts/tdnv/quick_tdnv_mmlu.sh -m Qwen/Qwen3-1.7B -q 200     # Custom model + questions
#   ./scripts/tdnv/quick_tdnv_mmlu.sh --categories math,physics      # Specific categories
#   ./scripts/tdnv/quick_tdnv_mmlu.sh --dry-run                      # Dry run
#
# Output:
#   JSON:  data/tdnv/mmlu_{model}.json
#   Plot:  plot/tdnv/mmlu_{model}.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Compute TDNV metrics across MMLU-Pro subject categories.

Each MMLU category (math, physics, chemistry, etc.) is treated as a separate
group. Lower TDNV = better separability between categories.

Options:
    -m, --model MODEL       HuggingFace model name (default: Qwen/Qwen3-1.7B)
    -q, --questions N       Number of MMLU questions (default: 100)
    --seed N                Random seed for question sampling (default: 42)
    --categories LIST       Comma-separated category names (default: all)
    -o, --output DIR        Output directory for JSON (default: data/tdnv)
    --plot-dir DIR          Output directory for plots (default: plot/tdnv)
    -l, --last-n N          Use only last N tokens (default: all tokens)
    --dry-run               Dry run (no model loading)
    -h, --help              Show this help

Examples:
    $(basename "$0")
    $(basename "$0") -m Qwen/Qwen3-1.7B -q 200
    $(basename "$0") --categories math,physics,chemistry
    $(basename "$0") --dry-run

EOF
    exit 0
}

MODEL="Qwen/Qwen3-1.7B"
NUM_QUESTIONS=100
MMLU_SEED=42
CATEGORIES=""
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv"
LAST_N_ARG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plot-dir)
            PLOT_DIR="$2"
            shift 2
            ;;
        -l|--last-n)
            LAST_N_ARG="--last-n $2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$PLOT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV MMLU Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Model:      ${GREEN}$MODEL${NC}"
echo -e "Questions:  ${GREEN}$NUM_QUESTIONS${NC}"
echo -e "Seed:       ${GREEN}$MMLU_SEED${NC}"
if [[ -n "$CATEGORIES" ]]; then
    echo -e "Categories: ${GREEN}$CATEGORIES${NC}"
else
    echo -e "Categories: ${GREEN}all${NC}"
fi
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Plots:      ${YELLOW}$PLOT_DIR${NC}"
echo -e "Dry run:    ${YELLOW}$DRY_RUN${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Build args array
TDNV_ARGS=(
    --mode mmlu
    --model "$MODEL"
    --num-questions "$NUM_QUESTIONS"
    --mmlu-seed "$MMLU_SEED"
    --output "$OUTPUT_DIR"
    --plot-dir "$PLOT_DIR"
)

if [[ -n "$CATEGORIES" ]]; then
    TDNV_ARGS+=(--categories "$CATEGORIES")
fi

if [[ -n "$LAST_N_ARG" ]]; then
    TDNV_ARGS+=($LAST_N_ARG)
fi

if [[ "$DRY_RUN" == true ]]; then
    TDNV_ARGS+=(--dry-run)
fi

echo -e "${GREEN}Running MMLU TDNV analysis...${NC}"
echo "----------------------------------------"

if uv run python -m "steering_geometry.tdnv" \
    "${TDNV_ARGS[@]}" 2>&1 | while read -r line; do
        echo "  $line"
    done; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✓ MMLU TDNV analysis complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "  JSON results: ${YELLOW}$OUTPUT_DIR${NC}"
    echo -e "  Plots:        ${YELLOW}$PLOT_DIR${NC}"
    echo -e "${GREEN}============================================${NC}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}✗ MMLU TDNV analysis failed${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
