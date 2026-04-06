#!/usr/bin/env bash
# =============================================================================
# run_discriminative_heatmaps.sh - Discriminative Token Selection Experiment
# =============================================================================
# Runs discriminative token selection experiments for all 5 concepts to generate
# cosine similarity heatmaps across varying K values and layers.
#
# Usage:
#   ./scripts/vector_analysis/run_discriminative_heatmaps.sh
#
# Concepts: honesty, sentiment, toxicity, sycophancy, refusal
# K values: [16, 32, 64, 128, 256]
# Layers: [0.4, 0.5, 0.6, 0.7, 0.8]
# Model: Qwen/Qwen3-1.7B
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load centralized configuration
eval $(uv run python -m steering_geometry --shell)

# Experiment configuration
K_VALUES=(16 32)
LAYERS=(0.4 0.5)
MODEL="Qwen/Qwen3-1.7B"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Convert arrays to Python list format
k_values_str=$(IFS=,; echo "[${K_VALUES[*]}]")
layers_str=$(IFS=,; echo "[${LAYERS[*]}]")

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Discriminative Token Selection Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:      ${GREEN}${ALL_CONCEPTS[*]}${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "K values:      ${GREEN}${k_values_str}${NC}"
echo -e "Layers:        ${GREEN}${layers_str}${NC}"
echo -e "Output dir:    ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run experiments sequentially
total=${#ALL_CONCEPTS[@]}
current=0

for concept in "${ALL_CONCEPTS[@]}"; do
    ((++current))
    echo -e "\n${GREEN}[$current/$total] Running discriminative experiment: $concept${NC}"
    echo "----------------------------------------"

    uv run python -u -c "
from steering_geometry.stability_comparison import run_discriminative_experiment
from pathlib import Path

result = run_discriminative_experiment(
    concept='${concept}',
    k_values=${k_values_str},
    layers=${layers_str},
    model_name='${MODEL}',
    output_dir=Path('${OUTPUT_DIR}')
)

print(f'  Vectors saved to: outputs/vectors/${concept}/discriminative/')
print(f'  Heatmaps saved to: outputs/heatmaps/discriminative/')

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
echo -e "  Heatmaps:         ${YELLOW}${OUTPUT_DIR}/heatmaps/discriminative/${NC}"
echo -e "${GREEN}============================================${NC}"
