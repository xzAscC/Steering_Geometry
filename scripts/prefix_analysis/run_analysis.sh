#!/usr/bin/env bash
# =============================================================================
# Prefix Steering Analysis
#
# Runs comprehensive analysis of why Prefix Steering works:
# 1. KL divergence: Prefix Steering vs No Steering
# 2. KL divergence: Prefix Steering vs All-Step Steering
# 3. Attention path analysis: how prefix steering affects attention
#
# Usage:
#   ./run_analysis.sh [concept] [model] [layer_frac] [steer_tokens] [num_prompts] [max_new_tokens]
#
# Examples:
#   ./run_analysis.sh                                    # defaults
#   ./run_analysis.sh sentiment Qwen/Qwen3-1.7B 0.7 10  # custom params
#   ./run_analysis.sh refusal Qwen/Qwen3-1.7B 0.7 15 10
#
# Output:
#   outputs/prefix_analysis/{concept}/{model}/
#     ├── plots/
#     │   ├── kl_prefix_vs_no_steer.pdf
#     │   ├── kl_prefix_vs_all_steer.pdf
#     │   ├── attention_to_prefix.pdf
#     │   └── attention_cosine_shift.pdf
#     └── analysis_report.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
CONCEPT="${1:-sentiment}"
MODEL="${2:-Qwen/Qwen3-1.7B}"
LAYER_FRAC="${3:-0.7}"
STEER_TOKENS="${4:-10}"
NUM_PROMPTS="${5:-10}"
MAX_NEW_TOKENS="${6:-100}"
SCALE_MULT="${7:-1.0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Prefix Steering Analysis ===${NC}"
echo -e "  Concept:      ${GREEN}${CONCEPT}${NC}"
echo -e "  Model:        ${GREEN}${MODEL}${NC}"
echo -e "  Layer:        ${GREEN}${LAYER_FRAC}${NC}"
echo -e "  Steer tokens: ${GREEN}${STEER_TOKENS}${NC}"
echo -e "  Num prompts:  ${GREEN}${NUM_PROMPTS}${NC}"
echo -e "  Max tokens:   ${GREEN}${MAX_NEW_TOKENS}${NC}"
echo -e "  Scale mult:   ${GREEN}${SCALE_MULT}${NC}"
echo ""

SAFE_MODEL="${MODEL//\//_}"
OUTPUT_DIR="outputs/prefix_analysis"

# Auto-discover vector path
VECTOR_PATH=""
for candidate in \
    "outputs/vectors/${CONCEPT}/discriminative/k128_layer${LAYER_FRAC}.pt" \
    "outputs/vectors/${CONCEPT}/diff_means/n6000_layer${LAYER_FRAC}.pt" \
    "outputs/vectors/${CONCEPT}/diff_means/n3000_layer${LAYER_FRAC}.pt"; do
    if [ -f "$candidate" ]; then
        VECTOR_PATH="$candidate"
        echo -e "Found vector: ${GREEN}${VECTOR_PATH}${NC}"
        break
    fi
done

VECTOR_ARG=""
if [ -n "$VECTOR_PATH" ]; then
    VECTOR_ARG="vector_path=Path('${VECTOR_PATH}'),"
fi

echo -e "${BLUE}Running analysis...${NC}"
echo ""

cd "$PROJECT_ROOT"
uv run python -u -c "
import logging
from pathlib import Path
from steering_geometry.utils import configure_logging
from steering_geometry.prefix_analysis import run_prefix_analysis

configure_logging(level='INFO')

report = run_prefix_analysis(
    model_name='${MODEL}',
    concept='${CONCEPT}',
    ${VECTOR_ARG}
    layer_frac=${LAYER_FRAC},
    steer_tokens=${STEER_TOKENS},
    scale_multiplier=${SCALE_MULT},
    num_prompts=${NUM_PROMPTS},
    max_new_tokens=${MAX_NEW_TOKENS},
    run_attention=True,
    output_dir=Path('${OUTPUT_DIR}'),
)

print()
print('=' * 60)
print('ANALYSIS COMPLETE')
print('=' * 60)
print(f'KL results: {len(report.kl_results)} prompts')
print(f'Attention results: {len(report.attention_results)} prompts')
print(f'Output dir: ${OUTPUT_DIR}')
"

echo ""
echo -e "${GREEN}=== Analysis Complete ===${NC}"
echo -e "Output directory: ${BLUE}${OUTPUT_DIR}${NC}"
echo -e "Report:           ${BLUE}${OUTPUT_DIR}/analysis_report.md${NC}"
echo -e "Plots:            ${BLUE}${OUTPUT_DIR}/plots/${NC}"
