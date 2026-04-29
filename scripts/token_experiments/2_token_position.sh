#!/usr/bin/env bash
# =============================================================================
# 2_token_position.sh - Token Position Experiment Runner
# =============================================================================
# Runs token position comparison experiments to investigate how different token
# selection strategies (all vs last_n with varying n) affect steering vectors.
#
# Usage:
#   ./scripts/token_experiments/2_token_position.sh [OPTIONS]
#
# Options:
#   -c, --concept      Concept to analyze (default: refusal)
#   -m, --model        Model name (default: Qwen/Qwen3-1.7B)
#   -n, --num-examples Number of contrast pairs (default: 100)
#   --last-n           Space-separated last_n values (default: "1 2 3 4 5 10")
#   --include-all      Include "all" mode: true/false (default: true)
#   -l, --layers       Layer fractions as space-separated list (default: 0.4 0.5 0.6 0.7 0.8)
#   -o, --output       Output directory (default: outputs/token_experiments)
#   -h, --help         Show this help message
#
# Example:
#   ./scripts/token_experiments/2_token_position.sh -c honesty -n 200
#   ./scripts/token_experiments/2_token_position.sh --last-n "1 2 4 8" --include-all false
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="refusal"
MODEL="Qwen/Qwen3-1.7B"
N_EXAMPLES=100
LAST_N_VALUES="1 2 3 4 5 10"
INCLUDE_ALL="true"
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
        -n|--num-examples)
            N_EXAMPLES="$2"
            shift 2
            ;;
        --last-n)
            LAST_N_VALUES="$2"
            shift 2
            ;;
        --include-all)
            INCLUDE_ALL="$2"
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
            echo "  -n, --num-examples Number of contrast pairs (default: 100)"
            echo "  --last-n           Space-separated last_n values (default: \"1 2 3 4 5 10\")"
            echo "  --include-all      Include \"all\" mode: true/false (default: true)"
            echo "  -l, --layers       Layer fractions as space-separated list"
            echo "  -o, --output       Output directory (default: outputs/token_experiments)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c honesty -n 200"
            echo "  $0 --last-n \"1 2 4 8\" --include-all false"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Convert space-separated strings to Python list format
layers_str="[${LAYERS// /, }]"
last_n_str="[${LAST_N_VALUES// /, }]"

# Determine include_all Python boolean
if [[ "$INCLUDE_ALL" == "true" ]]; then
    INCLUDE_ALL_PY="True"
else
    INCLUDE_ALL_PY="False"
fi

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Token Position Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Num examples:  ${GREEN}${N_EXAMPLES}${NC}"
echo -e "Last N values: ${GREEN}${LAST_N_VALUES}${NC}"
echo -e "Include all:   ${GREEN}${INCLUDE_ALL}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo -e "${YELLOW}Running token position experiment...${NC}"
echo ""

uv run python -u -c "
import json
from pathlib import Path

from steering_geometry.token_selection_experiments import run_token_position_experiment
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

# Build position_configs
position_configs = []
if ${INCLUDE_ALL_PY}:
    position_configs.append({'mode': 'all'})
for n in ${last_n_str}:
    position_configs.append({'mode': 'last_n', 'n': n})

if not position_configs:
    raise ValueError('No position configs generated. Enable --include-all or provide --last-n values.')

print(f'Position configs: {position_configs}')

# Run experiment
results = run_token_position_experiment(
    concept='${CONCEPT}',
    n_examples=${N_EXAMPLES},
    position_configs=position_configs,
    layers=${layers_str},
    model_name='${MODEL}',
    output_dir=Path('${OUTPUT_DIR}'),
    method='mean',
)

# Save summary
summary_path = Path('${OUTPUT_DIR}') / 'token_position_${CONCEPT}_summary.json'
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'Saved summary to: {summary_path}')

print('')
print('Experiment complete!')
print(f'Vectors saved to: ${OUTPUT_DIR}/vectors/${CONCEPT}/token_position/')
print(f'Heatmaps saved to: ${OUTPUT_DIR}/heatmaps/token_position/')
"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo -e "To view results:"
echo -e "  cat ${OUTPUT_DIR}/token_position_${CONCEPT}_summary.json"
echo -e "  ls ${OUTPUT_DIR}/vectors/${CONCEPT}/token_position/"
echo -e "  ls ${OUTPUT_DIR}/heatmaps/token_position/"
