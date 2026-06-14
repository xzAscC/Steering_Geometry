#!/bin/bash
# =============================================================================
# run_k_ablation.sh - Robust DiM K Ablation
# =============================================================================
# Varies the number of selected activations K with a fixed candidate pool.
# Paper figure: k_ablation_olmo3_7b.pdf
#
# Usage:
#   ./scripts/vector_analysis/run_k_ablation.sh [OPTIONS]
#
# Options:
#   -c, --concept        Concept (default: refusal)
#   -m, --model          Model name (default: allenai/Olmo-3-1025-7B)
#   -k, --k-values       Comma-separated K values (default: "10,20,30,50,100,128,200")
#   -l, --layers         Comma-separated layer fractions (default: "0.4,0.5,0.6,0.7,0.8")
#   -o, --output         Output directory (default: outputs/ablation/k_ablation)
#   -h, --help           Show this help message
#
# Example:
#   ./scripts/vector_analysis/run_k_ablation.sh -c sentiment -k "10,30,100,200"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="refusal"
MODEL="allenai/Olmo-3-1025-7B"
K_VALUES="10,20,30,50,100,128,200"
LAYERS="0.4,0.5,0.6,0.7,0.8"
OUTPUT_DIR="$PROJECT_ROOT/outputs/ablation/k_ablation"
LOG_LEVEL="INFO"

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
        -k|--k-values)
            K_VALUES="$2"
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
            echo "  -c, --concept        Concept (default: refusal)"
            echo "  -m, --model          Model name (default: allenai/Olmo-3-1025-7B)"
            echo "  -k, --k-values       Comma-separated K values (default: 10,20,30,50,100,128,200)"
            echo "  -l, --layers         Comma-separated layer fractions (default: 0.4,0.5,0.6,0.7,0.8)"
            echo "  -o, --output         Output directory (default: outputs/ablation/k_ablation)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c sentiment -k \"10,30,100,200\""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to Python list format
k_values_py="[${K_VALUES}]"
layers_py="[${LAYERS}]"

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Robust DiM K Ablation${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "K values:      ${GREEN}${K_VALUES}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo ""
echo -e "${YELLOW}Running K ablation experiment...${NC}"

uv run python -u -c "
from steering_geometry.stability_comparison import run_discriminative_experiment
from steering_geometry.utils import configure_logging

configure_logging(level='${LOG_LEVEL}')

result = run_discriminative_experiment(
    concept='${CONCEPT}',
    k_values=${k_values_py},
    layers=${layers_py},
    model_name='${MODEL}',
    output_dir='${OUTPUT_DIR}',
)
print('K ablation complete.')
print(f'Results: {len(result[\"statistics\"])} layer configurations')
"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
