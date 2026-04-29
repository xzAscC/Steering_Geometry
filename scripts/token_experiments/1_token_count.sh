#!/usr/bin/env bash
# =============================================================================
# 1_token_count.sh - Token Count Sweep Experiment
# =============================================================================
# Runs token count sweep experiment: extracts steering vectors with varying
# numbers of contrast pairs and computes pairwise cosine similarity.
#
# Usage:
#   ./scripts/token_experiments/1_token_count.sh [OPTIONS]
#
# Options:
#   -c, --concept      Concept to analyze (default: refusal)
#   -m, --model        Model name (default: Qwen/Qwen3-1.7B)
#   -n, --n-examples   Space-separated list of example counts (default: 10 30 100 300 1000 3000 6000 10000)
#   -l, --layers       Space-separated list of layer fractions (default: 0.4 0.5 0.6 0.7 0.8)
#   -o, --output       Output directory (default: outputs/token_experiments)
#   -h, --help         Show this help message
#
# Example:
#   ./scripts/token_experiments/1_token_count.sh -c honesty -n "10 100 1000"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="refusal"
MODEL="Qwen/Qwen3-1.7B"
N_EXAMPLES="10 30 100 300 1000 3000 6000 10000"
LAYERS="0.4 0.5 0.6 0.7 0.8"
OUTPUT_DIR="$PROJECT_ROOT/outputs/token_experiments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concept)
            CONCEPT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--n-examples)
            N_EXAMPLES="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --concept      Concept to analyze (default: refusal)"
            echo "  -m, --model        Model name (default: Qwen/Qwen3-1.7B)"
            echo "  -n, --n-examples   Space-separated list of example counts"
            echo "                     (default: 10 30 100 300 1000 3000 6000 10000)"
            echo "  -l, --layers       Space-separated list of layer fractions"
            echo "                     (default: 0.4 0.5 0.6 0.7 0.8)"
            echo "  -o, --output       Output directory (default: outputs/token_experiments)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c honesty -n \"10 100 1000\""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Convert space-separated strings to Python list format
n_examples_str="[${N_EXAMPLES// /, }]"
layers_str="[${LAYERS// /, }]"

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Token Count Sweep Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "N examples:    ${GREEN}${N_EXAMPLES}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo -e "${YELLOW}Running token count sweep experiment...${NC}"
echo ""

uv run python -u -c "
from steering_geometry.token_selection_experiments import run_token_count_experiment
from steering_geometry.utils import configure_logging

configure_logging(level="INFO")

# Run experiment
results = run_token_count_experiment(
    concept='${CONCEPT}',
    n_examples_list=${n_examples_str},
    layers=${layers_str},
    model_name='${MODEL}',
    output_dir='${OUTPUT_DIR}',
)

print('')
print('Experiment complete!')
print(f'Results saved to: ${OUTPUT_DIR}')
"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo -e "Output structure:"
echo -e "  ${OUTPUT_DIR}/vectors/${CONCEPT}/  - Steering vectors per example count"
echo -e "  ${OUTPUT_DIR}/heatmaps/            - Cosine similarity heatmaps"
