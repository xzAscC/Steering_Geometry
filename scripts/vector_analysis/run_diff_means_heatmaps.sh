#!/usr/bin/env bash
# =============================================================================
# run_diff_means_heatmaps.sh - Differential Means Experiment Runner
# =============================================================================
# Runs differential means experiments for all 5 concepts to generate
# cosine similarity heatmaps across varying example counts and layers.
#
# Usage:
#   ./scripts/experiments/run_diff_means_heatmaps.sh
#
# Concepts: honesty, sentiment, toxicity, sycophancy, refusal
# Example counts: [10, 30, 100, 300, 1000, 3000, 6000, 10000] (capped per concept)
# Layers: [0.4, 0.5, 0.6, 0.7, 0.8]
# Model: Qwen/Qwen3-1.7B
#
# Dataset limits (from Task 0 validation):
#   - honesty: 800
#   - sycophancy: 4000
#   - refusal: 1000
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Experiment configuration
CONCEPTS=("honesty" "sentiment" "toxicity" "sycophancy" "refusal")
N_EXAMPLES=(10 30 100 300 1000 3000 6000 10000)
LAYERS=(0.4 0.5 0.6 0.7 0.8)
MODEL="Qwen/Qwen3-1.7B"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Convert arrays to Python list format
n_examples_str=$(IFS=,; echo "[${N_EXAMPLES[*]}]")
layers_str=$(IFS=,; echo "[${LAYERS[*]}]")

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Differential Means Experiment Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:      ${GREEN}${CONCEPTS[*]}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "N examples:    ${GREEN}${n_examples_str}${NC}"
echo -e "Layers:        ${GREEN}${layers_str}${NC}"
echo -e "Output dir:    ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run experiments sequentially
total=${#CONCEPTS[@]}
current=0

for concept in "${CONCEPTS[@]}"; do
    ((++current))
    echo -e "\n${GREEN}[$current/$total] Running diff_means experiment: $concept${NC}"
    echo "----------------------------------------"

    uv run python -u -c "
from steering_geometry.vector_analysis import run_diff_means_experiment
from pathlib import Path

result = run_diff_means_experiment(
    concept='${concept}',
    n_examples_list=${n_examples_str},
    layers=${layers_str},
    model_name='${MODEL}',
    output_dir=Path('${OUTPUT_DIR}')
)

print(f'  Vectors saved to: outputs/vectors/${concept}/diff_means/')
print(f'  Heatmaps saved to: outputs/heatmaps/diff_means/')

# Print statistics
stats = result.get('statistics', {})
if stats:
    print(f'  Statistics:')
    for layer, layer_stats in stats.items():
        m = layer_stats.get('mean_similarity', layer_stats.get('mean', 'N/A'))
        mn = layer_stats.get('min_similarity', layer_stats.get('min', 'N/A'))
        mx = layer_stats.get('max_similarity', layer_stats.get('max', 'N/A'))
        print(f'    Layer {layer}: mean={m:.4f}, min={mn:.4f}, max={mx:.4f}')
" 2>&1

    echo -e "${GREEN}✓ Completed: $concept${NC}"
done

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}✓ All experiments complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Output directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Vectors:          ${YELLOW}${OUTPUT_DIR}/vectors/${NC}"
echo -e "  Heatmaps:         ${YELLOW}${OUTPUT_DIR}/heatmaps/diff_means/${NC}"
echo -e "${GREEN}============================================${NC}"
