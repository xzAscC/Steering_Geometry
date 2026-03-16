#!/usr/bin/env bash
# =============================================================================
# run_discriminative.sh - Discriminative Token Steering Vector Extraction
# =============================================================================
# Extracts steering vectors using the discriminative token selection method.
#
# Discriminative Method:
#   - Selects top-k tokens that best discriminate between classes
#   - Score: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
#   - Higher scores = closer to own class, farther from other class
#   - Steering direction: v = μ_+^disc - μ_-^disc
#
# Usage:
#   ./scripts/run_discriminative.sh                                    # All concepts, default model
#   ./scripts/run_discriminative.sh -c honesty,toxicity                # Specific concepts
#   ./scripts/run_discriminative.sh -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
#   ./scripts/run_discriminative.sh --top-k 50                         # Custom top-k value
#
# Output:
#   data/vectors/{concept}_{model}_discriminative.pt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

Extract steering vectors using the DISCRIMINATIVE token selection method.

The discriminative method selects the top-k tokens that best separate
positive and negative classes, focusing on the most informative activations.

Scoring Formula:
  - Score: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
  - Higher score = token is closer to own class center
  - Lower score = token is closer to other class center
  - Select top-k tokens per class and compute prototype vectors
  - Steering direction: v = μ_+^disc - μ_-^disc

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    -p, --pairs N          Number of contrast pairs (default: 500)
    -k, --top-k N          Number of top tokens to select (default: 100)
    -o, --output DIR       Output directory (default: data/vectors)
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts, default model
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
    $(basename "$0") -c sentiment -p 100 --top-k 50    # Custom params
    $(basename "$0") -c all -m all                      # All concepts × all models

Output Files:
    data/vectors/{concept}_{model}_discriminative.pt

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
TOP_K=100
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
        -k|--top-k)
            TOP_K="$2"
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
echo -e "${BLUE}Discriminative Token Extraction Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Method:    ${GREEN}discriminative${NC}"
echo -e "Concepts:  ${GREEN}${CONCEPT_ARRAY[*]}${NC}"
echo -e "Models:    ${GREEN}${MODEL_ARRAY[*]}${NC}"
echo -e "Pairs:     ${GREEN}$NUM_PAIRS${NC}"
echo -e "Top-k:     ${GREEN}$TOP_K${NC}"
echo -e "Output:    ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Total:     ${GREEN}$total extraction(s)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Method Details:"
echo -e "  ${GREEN}•${NC} Selects top-k most discriminative tokens"
echo -e "  ${GREEN}•${NC} Focuses on informative activations"
echo -e "  ${GREEN}•${NC} Reduces noise from uninformative tokens"
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
            --method "discriminative" \
            --top-k "$TOP_K" \
            --output "$OUTPUT_DIR" 2>&1 | while read -r line; do
                echo "  $line"
            done
        
        echo ""
    done
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ All discriminative extractions complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Vectors saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "${GREEN}============================================${NC}"
