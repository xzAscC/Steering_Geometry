#!/usr/bin/env bash
# =============================================================================
# run_tdnv.sh - TDNV (Topic-Discriminative Normalized Variance) Analysis
# =============================================================================
# Computes separability metrics for positive/negative contrast pairs across
# all model layers to analyze steering vector effectiveness.
#
# Outputs three TDNV-related values:
#   1. TDNV      - Topic-Discriminative Normalized Variance (lower = better separability)
#   2. NormNum   - Normalized within-topic variance
#   3. NormDen   - Normalized between-topic distance
#   (+ Layerwise Energy)
#
# Usage:
#   ./scripts/run_tdnv.sh                                    # All concepts, default model
#   ./scripts/run_tdnv.sh -c honesty,toxicity                # Specific concepts
#   ./scripts/run_tdnv.sh -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
#   ./scripts/run_tdnv.sh -c honesty --dry-run               # Dry run (no model loading)
#
# Output:
#   - JSON: data/tdnv/{concept}_{model}.json
#   - Plot: plot/tdnv/{concept}_{model}.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ALL_CONCEPTS=("honesty" "sycophancy" "toxicity" "sentiment" "refusal")
ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "google/gemma-2-2b")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Compute TDNV (Topic-Discriminative Normalized Variance) metrics for steering analysis.

TDNV measures separability between positive and negative activations.
Lower TDNV = better separability = easier to steer.

Output Values:
  - TDNV      : Topic-Discriminative Normalized Variance
  - NormNum   : Normalized within-topic variance
  - NormDen   : Normalized between-topic distance
  - Energy    : Layerwise activation energy

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    -p, --pairs N          Number of contrast pairs (default: 500)
    -o, --output DIR       Output directory for JSON results (default: data/tdnv)
    --plot-dir DIR         Output directory for plots (default: plot/tdnv)
    --dry-run              Load data only, skip model loading and computation
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts, default model
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
    $(basename "$0") -c sentiment -p 100                # Custom pairs
    $(basename "$0") -c all -m all                      # All concepts × all models
    $(basename "$0") -c honesty --dry-run               # Dry run

Output Files:
    JSON:  data/tdnv/{concept}_{model}.json
    Plot:  plot/tdnv/{concept}_{model}.pdf

EOF
    exit 0
}

list_available() {
    echo "Available concepts:"
    printf '  - %s\n' "${ALL_CONCEPTS[@]}"
    echo ""
    echo "Available models:"
    printf '  - %s\n' "${ALL_MODELS[@]}"
    exit 0
}

CONCEPTS=""
MODELS="Qwen/Qwen3.5-2B"
NUM_PAIRS=500
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concepts)
            CONCEPTS="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -p|--pairs)
            NUM_PAIRS="$2"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -l|--list)
            list_available
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

IFS=',' read -ra CONCEPT_ARRAY <<< "$CONCEPTS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

if [[ "$CONCEPTS" == "all" || -z "$CONCEPTS" ]]; then
    CONCEPT_ARRAY=("${ALL_CONCEPTS[@]}")
fi

if [[ "$MODELS" == "all" ]]; then
    MODEL_ARRAY=("${ALL_MODELS[@]}")
fi

mkdir -p "$OUTPUT_DIR" "$PLOT_DIR"

total=$((${#CONCEPT_ARRAY[@]} * ${#MODEL_ARRAY[@]}))
current=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV Analysis Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:   ${GREEN}${CONCEPT_ARRAY[*]}${NC}"
echo -e "Models:     ${GREEN}${MODEL_ARRAY[*]}${NC}"
echo -e "Pairs:      ${GREEN}$NUM_PAIRS${NC}"
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Plots:      ${YELLOW}$PLOT_DIR${NC}"
echo -e "Dry run:    ${YELLOW}$DRY_RUN${NC}"
echo -e "Total:      ${GREEN}$total analysis(es)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Computing 3 TDNV-related values:"
echo -e "  ${GREEN}1. TDNV${NC}     - Topic-Discriminative Normalized Variance"
echo -e "  ${GREEN}2. NormNum${NC}  - Normalized within-topic variance"
echo -e "  ${GREEN}3. NormDen${NC}  - Normalized between-topic distance"
echo ""

DRY_RUN_FLAG=""
if [[ "$DRY_RUN" == true ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

for model in "${MODEL_ARRAY[@]}"; do
    for concept in "${CONCEPT_ARRAY[@]}"; do
        ((current++))
        echo -e "${GREEN}[$current/$total] Analyzing: $concept × $model${NC}"
        echo "----------------------------------------"
        
        uv run python -m "steering_geometry.tdnv" \
            --concept "$concept" \
            --model "$model" \
            --num-pairs "$NUM_PAIRS" \
            --output "$OUTPUT_DIR" \
            --plot-dir "$PLOT_DIR" \
            $DRY_RUN_FLAG 2>&1 | while read -r line; do
                echo "  $line"
            done
        
        echo ""
    done
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ TDNV analysis complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  JSON results: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  Plots:        ${YELLOW}$PLOT_DIR${NC}"
echo -e "${GREEN}============================================${NC}"
