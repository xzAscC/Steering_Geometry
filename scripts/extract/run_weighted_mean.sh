#!/usr/bin/env bash
# =============================================================================
# run_weighted_mean.sh - Weighted Mean Steering Vector Extraction
# =============================================================================
# Extracts steering vectors using the weighted_mean method.
#
# Weighted Mean Method:
#   - Computes distance-based weights for tokens relative to class center
#   - Tokens closer to class center receive larger weights
#   - Formula: w_i = exp(-||h_i - h̄||² / τ²)
#   - Steering direction: v = μ_+^w - μ_-^w
#
# Usage:
#   ./scripts/extract/run_weighted_mean.sh                                    # All concepts, default model
#   ./scripts/extract/run_weighted_mean.sh -c honesty,toxicity                # Specific concepts
#   ./scripts/extract/run_weighted_mean.sh -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
#
# Output:
#   data/vectors/{concept}_{model}_weighted_mean.pt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ALL_CONCEPTS=("honesty" "sycophancy" "toxicity" "sentiment" "refusal")
ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "google/gemma-2-2b")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Extract steering vectors using the WEIGHTED MEAN method.

The weighted mean method computes distance-based weights for tokens,
giving larger weights to tokens closer to the class center. This reduces
the influence of outliers and noise in the extraction process.

Formula:
  - Center: h̄_c = (1/n_c) Σ h_i^(c)
  - Variance: τ_c² = (1/n_c) Σ ||h_i^(c) - h̄_c||²
  - Weights: w_i^(c) = exp(-||h_i^(c) - h̄_c||² / τ_c²)
  - Weighted mean: μ_c^w = Σ w_i^(c) h_i^(c) / Σ w_i^(c)
  - Steering direction: v = μ_+^w - μ_-^w

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    -p, --pairs N          Number of contrast pairs (default: 500)
    -o, --output DIR       Output directory (default: data/vectors)
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts, default model
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
    $(basename "$0") -c sentiment -p 100                # Custom params
    $(basename "$0") -c all -m all                      # All concepts × all models

Output Files:
    data/vectors/{concept}_{model}_weighted_mean.pt

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

# Default values
CONCEPTS=""
MODELS="Qwen/Qwen3.5-2B"
NUM_PAIRS=500
OUTPUT_DIR="$PROJECT_ROOT/data/vectors"

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

mkdir -p "$OUTPUT_DIR"

total=$((${#CONCEPT_ARRAY[@]} * ${#MODEL_ARRAY[@]}))
current=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Weighted Mean Extraction Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Method:    ${GREEN}weighted_mean${NC}"
echo -e "Concepts:  ${GREEN}${CONCEPT_ARRAY[*]}${NC}"
echo -e "Models:    ${GREEN}${MODEL_ARRAY[*]}${NC}"
echo -e "Pairs:     ${GREEN}$NUM_PAIRS${NC}"
echo -e "Output:    ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Total:     ${GREEN}$total extraction(s)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Method Details:"
echo -e "  ${GREEN}•${NC} Distance-based weighting"
echo -e "  ${GREEN}•${NC} Reduces outlier influence"
echo -e "  ${GREEN}•${NC} Robust to noise"
echo ""

for model in "${MODEL_ARRAY[@]}"; do
    for concept in "${CONCEPT_ARRAY[@]}"; do
        ((current++))
        echo -e "${GREEN}[$current/$total] Extracting: $concept × $model${NC}"
        echo "----------------------------------------"
        
        uv run python -m "steering_geometry.extract" \
            --concept "$concept" \
            --model "$model" \
            --num-pairs "$NUM_PAIRS" \
            --method "weighted_mean" \
            --output "$OUTPUT_DIR" 2>&1 | while read -r line; do
                echo "  $line"
            done
        
        echo ""
    done
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ All weighted_mean extractions complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Vectors saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "${GREEN}============================================${NC}"
