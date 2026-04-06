#!/usr/bin/env bash
# =============================================================================
# run_unembed_analysis.sh - Batch Unembedding Analysis
# =============================================================================
# Runs unembedding analysis across all concept × method combinations
#
# Usage:
#   ./scripts/unembed_analysis/run_unembed_analysis.sh                          # All concepts × all methods
#   ./scripts/unembed_analysis/run_unembed_analysis.sh -c honesty,toxicity      # Specific concepts
#   ./scripts/unembed_analysis/run_unembed_analysis.sh -m "Qwen/Qwen3.5-2B"     # Specific model
#   ./scripts/unembed_analysis/run_unembed_analysis.sh -l "0.5,0.6,0.7"         # Specific layers
#
# Analysis Matrix:
#   Concepts:  honesty, sentiment, toxicity, sycophancy, refusal
#   Methods:   diff_means, discriminative
#   Total:     10 combinations (5 concepts × 2 methods)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Available options
eval $(uv run python -m steering_geometry --shell)
ALL_METHODS=("diff_means" "discriminative")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run unembedding analysis across concepts and methods.

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sentiment, toxicity, sycophancy, refusal
    -m, --model MODEL      Model to analyze (default: Qwen/Qwen3-1.7B)
    -l, --layers LIST      Comma-separated layer fractions (default: 0.1,0.2,...,1.0)
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts × all methods
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m "Qwen/Qwen3.5-2B"               # Different model
    $(basename "$0") -l "0.5,0.6,0.7,0.8"               # Specific layers

EOF
    exit 0
}

# Default values
CONCEPTS=""
MODEL="Qwen/Qwen3-1.7B"
LAYERS="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concepts)
            CONCEPTS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
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

# Parse concepts
IFS=',' read -ra CONCEPT_ARRAY <<< "$CONCEPTS"

if [[ "$CONCEPTS" == "all" || -z "$CONCEPTS" ]]; then
    CONCEPT_ARRAY=("${ALL_CONCEPTS[@]}")
fi

# Calculate total runs
total=$((${#CONCEPT_ARRAY[@]} * ${#ALL_METHODS[@]}))

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Unembedding Analysis Batch${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:     ${GREEN}${CONCEPT_ARRAY[*]}${NC}"
echo -e "Methods:      ${GREEN}${ALL_METHODS[*]}${NC}"
echo -e "Model:        ${GREEN}${MODEL}${NC}"
echo -e "Layers:       ${GREEN}${LAYERS}${NC}"
echo -e "Total runs:   ${GREEN}$total${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Run analyses
current=0
for concept in "${CONCEPT_ARRAY[@]}"; do
    for method in "${ALL_METHODS[@]}"; do
        ((++current))
        echo -e "\n${GREEN}[$current/$total] Analyzing: $concept × $method${NC}"
        echo "----------------------------------------"
        
        uv run python -m steering_geometry.unembed_analysis \
            --concept "$concept" \
            --method "$method" \
            --model "$MODEL" \
            --layers "$LAYERS" 2>&1 | while read -r line; do
                echo "  $line"
            done
    done
done

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Batch analysis complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Completed: ${GREEN}$total${NC} analyses"
echo -e "${GREEN}============================================${NC}"
