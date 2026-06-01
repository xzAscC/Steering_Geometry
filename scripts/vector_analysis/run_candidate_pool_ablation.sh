#!/bin/bash
# =============================================================================
# run_candidate_pool_ablation.sh - Robust DiM Candidate Pool Ablation
# =============================================================================
# Varies candidate pool size (num_pairs) with fixed top_k.
# Paper figure: ncand_ablation_olmo3_7b.pdf
#
# Usage:
#   ./scripts/vector_analysis/run_candidate_pool_ablation.sh [OPTIONS]
#
# Options:
#   -c, --concept        Concept (default: safety)
#   -m, --model          Model name (default: allenai/Olmo-3-1025-7B)
#   -p, --pool-sizes     Comma-separated pool sizes (default: "50,100,200,500,1000")
#   -k, --top-k          Fixed top_k value (default: 50)
#   -l, --layers         Comma-separated layer fractions (default: "0.4,0.5,0.6,0.7,0.8")
#   -n, --num-trials     Number of trials per pool size (default: 3)
#   -o, --output         Output directory (default: outputs/ablation/candidate_pool)
#   -h, --help           Show this help message
#
# Example:
#   ./scripts/vector_analysis/run_candidate_pool_ablation.sh -c sentiment -p "50,100,500"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="safety"
MODEL="allenai/Olmo-3-1025-7B"
POOL_SIZES="50,100,200,500,1000"
TOP_K=50
LAYERS="0.4,0.5,0.6,0.7,0.8"
NUM_TRIALS=3
OUTPUT_DIR="$PROJECT_ROOT/outputs/ablation/candidate_pool"
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
        -p|--pool-sizes)
            POOL_SIZES="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        -n|--num-trials)
            NUM_TRIALS="$2"
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
            echo "  -c, --concept        Concept (default: safety)"
            echo "  -m, --model          Model name (default: allenai/Olmo-3-1025-7B)"
            echo "  -p, --pool-sizes     Comma-separated pool sizes (default: 50,100,200,500,1000)"
            echo "  -k, --top-k          Fixed top_k value (default: 50)"
            echo "  -l, --layers         Comma-separated layer fractions (default: 0.4,0.5,0.6,0.7,0.8)"
            echo "  -n, --num-trials     Number of trials per pool size (default: 3)"
            echo "  -o, --output         Output directory (default: outputs/ablation/candidate_pool)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c sentiment -p \"50,100,500\""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to Python list format
pool_sizes_py="[${POOL_SIZES}]"
layers_py="[${LAYERS}]"

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Robust DiM Candidate Pool Ablation${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Pool sizes:    ${GREEN}${POOL_SIZES}${NC}"
echo -e "Top K:         ${GREEN}${TOP_K}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Num trials:    ${GREEN}${NUM_TRIALS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo ""
echo -e "${YELLOW}Running candidate pool ablation experiment...${NC}"

uv run python -u -c "
from steering_geometry.stability_comparison import run_candidate_pool_ablation
from steering_geometry.utils import configure_logging

configure_logging(level='${LOG_LEVEL}')

result = run_candidate_pool_ablation(
    concept='${CONCEPT}',
    pool_sizes=${pool_sizes_py},
    top_k=${TOP_K},
    layers=${layers_py},
    model_name='${MODEL}',
    output_dir='${OUTPUT_DIR}',
    num_trials=${NUM_TRIALS},
)
print('Candidate pool ablation complete.')
print(f'Results: {len(result[\"statistics\"])} pool size configurations')
"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
