#!/bin/bash
# =============================================================================
# run_stability.sh - Stability Comparison Experiment Runner
# =============================================================================
# Runs stability comparison experiments to compare diff_means vs discriminative
# extraction methods by computing pairwise cosine similarity across runs.
#
# Usage:
#   ./scripts/stability_comparison/run_stability.sh [OPTIONS]
#
# Options:
#   -c, --concept      Concept to analyze (default: sentiment)
#   -n, --num-tokens   Number of tokens per run (default: 10000)
#   -r, --num-runs     Number of runs for comparison (default: 3)
#   -l, --layers       Layer fractions as space-separated list (default: 0.0 0.1 ... 0.9)
#   -k, --top-k        Top K for discriminative method (default: 30)
#   -m, --model        Model name (default: Qwen/Qwen3-1.7B)
#   -o, --output       Output directory (default: outputs/stability)
#   -h, --help         Show this help message
#
# Example:
#   ./scripts/stability_comparison/run_stability.sh -c refusal -n 1000 -r 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="sentiment"
NUM_TOKENS=1000
NUM_RUNS=3
LAYERS="0.0 0.5 0.9"
TOP_K=30
MODEL="Qwen/Qwen3-1.7B"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stability"

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
        -n|--num-tokens)
            NUM_TOKENS="$2"
            shift 2
            ;;
        -r|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
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
            echo "  -c, --concept      Concept to analyze (default: sentiment)"
            echo "  -n, --num-tokens   Number of tokens per run (default: 10000)"
            echo "  -r, --num-runs     Number of runs for comparison (default: 3)"
            echo "  -l, --layers       Layer fractions as space-separated list"
            echo "  -k, --top-k        Top K for discriminative method (default: 30)"
            echo "  -m, --model        Model name (default: Qwen/Qwen3-1.7B)"
            echo "  -o, --output       Output directory (default: outputs/stability)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c refusal -n 1000 -r 3"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Convert layers string to Python list format (comma-separated)
layers_str="[${LAYERS// /, }]"

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Stability Comparison Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:       ${GREEN}${CONCEPT}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Num tokens:    ${GREEN}${NUM_TOKENS}${NC}"
echo -e "Num runs:      ${GREEN}${NUM_RUNS}${NC}"
echo -e "Top K:         ${GREEN}${TOP_K}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiment
echo -e "${YELLOW}Running stability comparison experiment...${NC}"
echo ""

uv run python -u -c "
import json
from pathlib import Path

from steering_geometry.config import StabilityComparisonConfig
from steering_geometry.stability_comparison import (
    generate_stability_heatmap,
    run_stability_comparison_experiment,
    save_results_json,
)
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

# Configuration
config = StabilityComparisonConfig(
    concept='${CONCEPT}',
    num_tokens=${NUM_TOKENS},
    num_runs=${NUM_RUNS},
    layers=${layers_str},
    top_k=${TOP_K},
    model_name='${MODEL}',
    output_dir=Path('${OUTPUT_DIR}'),
)

# Run experiment
results = run_stability_comparison_experiment(config)

# Save JSON results
output_path = Path('${OUTPUT_DIR}') / 'stability_comparison_${CONCEPT}.json'
save_results_json(results, output_path)
print(f'Saved JSON results to: {output_path}')

# Generate heatmaps for each layer and method
heatmap_dir = Path('${OUTPUT_DIR}') / 'heatmaps'
for method in ['diff_means', 'discriminative']:
    for layer_str, sim_matrix in results[method]['similarity_matrices'].items():
        layer = float(layer_str)
        safe_method = method.replace('_', '-')
        output_path = heatmap_dir / f'{safe_method}_layer_{layer:.2f}.pdf'
        generate_stability_heatmap(sim_matrix, layer, method, output_path)
        print(f'Generated heatmap: {output_path}')

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
echo -e "To view results:"
echo -e "  cat ${OUTPUT_DIR}/stability_comparison_${CONCEPT}.json"
echo -e "  ls ${OUTPUT_DIR}/heatmaps/"
