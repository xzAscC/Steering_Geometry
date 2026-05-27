#!/usr/bin/env bash
# =============================================================================
# 4_steering_scope.sh - Steering Scope Experiment
# =============================================================================
# Runs steering scope experiment: applies steering with varying numbers of
# steered generation tokens, layers, and multipliers.
#
# Usage:
#   ./scripts/token_experiments/4_steering_scope.sh [OPTIONS]
#
# Options:
#   -c, --concept        Concept to analyze (default: refusal)
#   -m, --model          Model name (default: Qwen/Qwen3-1.7B)
#   --steer-tokens       Space-separated list of steer_tokens values (default: "5 10 15 20")
#   --include-full       Include None (all tokens) in steer_tokens list: true/false (default: true)
#   -l, --layers         Space-separated list of layer fractions (default: 0.4 0.5 0.6 0.7 0.8)
#   --multipliers        Space-separated list of multipliers (default: "0.01 0.1 1.0 10.0")
#   -n, --num-samples    Number of prompt samples (default: 10)
#   -o, --output         Output directory (default: outputs/token_experiments)
#   -h, --help           Show this help message
#
# Example:
#   ./scripts/token_experiments/4_steering_scope.sh -c refusal --steer-tokens "5 10 15 20"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="refusal"
MODEL="Qwen/Qwen3-1.7B"
STEER_TOKENS_VALUES="5 10 15 20"
INCLUDE_FULL="true"
LAYERS="0.4 0.5 0.6 0.7 0.8"
MULTIPLIERS="0.01 0.1 1.0 10.0"
NUM_SAMPLES=10
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
        --steer-tokens)
            STEER_TOKENS_VALUES="$2"
            shift 2
            ;;
        --include-full)
            INCLUDE_FULL="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        --multipliers)
            MULTIPLIERS="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
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
            echo "  -c, --concept        Concept to analyze (default: refusal)"
            echo "  -m, --model          Model name (default: Qwen/Qwen3-1.7B)"
            echo "  --steer-tokens       Space-separated list of steer_tokens values"
            echo "                        (default: \"5 10 15 20\")"
            echo "  --include-full       Include None (all tokens): true/false (default: true)"
            echo "  -l, --layers         Space-separated list of layer fractions"
            echo "                        (default: 0.4 0.5 0.6 0.7 0.8)"
            echo "  --multipliers        Space-separated list of multipliers"
            echo "                        (default: \"0.01 0.1 1.0 10.0\")"
            echo "  -n, --num-samples    Number of prompt samples (default: 10)"
            echo "  -o, --output         Output directory (default: outputs/token_experiments)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c refusal --steer-tokens \"5 10 15 20\""
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
multipliers_str="[${MULTIPLIERS// /, }]"

# Build steer_tokens Python list: optionally prepend None
if [[ "$INCLUDE_FULL" == "true" ]]; then
    steer_tokens_str="[None, ${STEER_TOKENS_VALUES// /, }]"
else
    steer_tokens_str="[${STEER_TOKENS_VALUES// /, }]"
fi

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Steering Scope Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Steer tokens:  ${GREEN}${STEER_TOKENS_VALUES}${NC}"
echo -e "Include full:  ${GREEN}${INCLUDE_FULL}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Multipliers:   ${GREEN}${MULTIPLIERS}${NC}"
echo -e "Num samples:   ${GREEN}${NUM_SAMPLES}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo -e "${YELLOW}Running steering scope experiment...${NC}"
echo ""

uv run python -u -c "
from steering_geometry.token_selection_experiments import run_steering_scope_experiment
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

# Run experiment
results = run_steering_scope_experiment(
    concept='${CONCEPT}',
    steer_tokens_values=${steer_tokens_str},
    layers=${layers_str},
    multipliers=${multipliers_str},
    model_name='${MODEL}',
    output_dir='${OUTPUT_DIR}',
    num_samples=${NUM_SAMPLES},
)

print('')
print('Experiment complete!')
print(f'Total samples generated: {results[\"statistics\"][\"total_samples\"]}')
print(f'Output files: {len(results[\"output_files\"])}')
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
echo -e "  ${OUTPUT_DIR}/steered/${CONCEPT}/steer_scope/  - JSONL results per parameter combo"
