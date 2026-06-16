#!/usr/bin/env bash
# =============================================================================
# Prefix Steering — KL Divergence Analysis (KL-only)
#
# Runs ONLY the KL divergence experiments for Prefix Steering, skipping the
# expensive attention-pattern analysis (which requires loading a second model
# with eager attention). Use this when you only need the KL curves and the
# prefix-length sweep.
#
# Produces:
# 1. Per-step KL:     KL(Prefix Steering ‖ No Steering)   over generation steps
# 2. Per-step KL:     KL(Prefix Steering ‖ All-Step Steering) over generation steps
# 3. Prefix-length sweep: KL averaged over the first K post-steer steps vs N
#
# Usage:
#   ./run_kl_divergence.sh [concept] [model] [layer_frac] [steer_tokens] [num_prompts] [max_new_tokens] [scale_mult] [steer_tokens_list] [num_post_steer_steps]
#
# Examples:
#   ./run_kl_divergence.sh                                         # defaults
#   ./run_kl_divergence.sh sentiment Qwen/Qwen3-1.7B 0.7 10 10
#   ./run_kl_divergence.sh refusal Qwen/Qwen3-1.7B 0.7 15 10
#   ./run_kl_divergence.sh sentiment Qwen/Qwen3-1.7B 0.7 10 10 100 0.1 "0,4,8,12" 3
#
# Args 8-9 (optional):
#   steer_tokens_list      Comma-separated prefix lengths for the KL sweep
#                          (e.g. "0,2,4,6,8,10"). Empty -> Python default.
#   num_post_steer_steps   Unsteered steps observed after steering ends (int).
#                          Empty -> Python default.
#
# Output:
#   outputs/prefix_analysis/{concept}/{model}/
#     ├── kl_prefix_vs_no_steer.pdf
#     ├── kl_prefix_vs_all_steer.pdf
#     ├── kl_prefix_length_sweep.pdf
#     └── analysis_report.md        (KL sections populated; attention sections empty)
#
# Note: For the full analysis including attention patterns, use run_analysis.sh instead.
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
SCALE_MULT="${7:-0.1}"
STEER_TOKENS_LIST_STR="${8:-}"
NUM_POST_STEER_STEPS_STR="${9:-}"

# Validate optional args
if [ -n "$STEER_TOKENS_LIST_STR" ] \
    && ! [[ "$STEER_TOKENS_LIST_STR" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: STEER_TOKENS_LIST must be comma-separated ints (got: '$STEER_TOKENS_LIST_STR')" >&2
    exit 1
fi
if [ -n "$NUM_POST_STEER_STEPS_STR" ] \
    && ! [[ "$NUM_POST_STEER_STEPS_STR" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_POST_STEER_STEPS must be a non-negative int (got: '$NUM_POST_STEER_STEPS_STR')" >&2
    exit 1
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Prefix Steering KL Divergence Analysis (KL-only) ===${NC}"
echo -e "  Concept:      ${GREEN}${CONCEPT}${NC}"
echo -e "  Model:        ${GREEN}${MODEL}${NC}"
echo -e "  Layer:        ${GREEN}${LAYER_FRAC}${NC}"
echo -e "  Steer tokens: ${GREEN}${STEER_TOKENS}${NC}"
echo -e "  Num prompts:  ${GREEN}${NUM_PROMPTS}${NC}"
echo -e "  Max tokens:   ${GREEN}${MAX_NEW_TOKENS}${NC}"
echo -e "  Scale mult:   ${GREEN}${SCALE_MULT}${NC}"
STEER_TOKENS_LIST_DISPLAY="${STEER_TOKENS_LIST_STR:-<default>}"
NUM_POST_STEER_STEPS_DISPLAY="${NUM_POST_STEER_STEPS_STR:-<default>}"
echo -e "  Steer list:   ${GREEN}${STEER_TOKENS_LIST_DISPLAY}${NC}"
echo -e "  Post steps:   ${GREEN}${NUM_POST_STEER_STEPS_DISPLAY}${NC}"
echo -e "  Attention:    ${RED}skipped${NC} (KL-only run)"
echo ""

SAFE_MODEL="${MODEL//\//_}"
OUTPUT_DIR="outputs/prefix_analysis"

# Auto-discover vector path (discriminative first, then diff_means fallbacks)
# TODO: add more layers, models and more vectors 
# TODO: change hyper-para here
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

STEER_TOKENS_LIST_ARG=""
if [ -n "$STEER_TOKENS_LIST_STR" ]; then
    # shellcheck disable=SC2086
    STEER_TOKENS_LIST_ARG="steer_tokens_list=[int(x) for x in '${STEER_TOKENS_LIST_STR}'.split(',')],"
fi

NUM_POST_STEER_STEPS_ARG=""
if [ -n "$NUM_POST_STEER_STEPS_STR" ]; then
    NUM_POST_STEER_STEPS_ARG="num_post_steer_steps=${NUM_POST_STEER_STEPS_STR},"
fi

echo -e "${BLUE}Running KL divergence analysis...${NC}"
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
    run_attention=False,
    ${STEER_TOKENS_LIST_ARG}
    ${NUM_POST_STEER_STEPS_ARG}
    output_dir=Path('${OUTPUT_DIR}'),
)

print()
print('=' * 60)
print('KL DIVERGENCE ANALYSIS COMPLETE')
print('=' * 60)
print(f'KL results:        {len(report.kl_results)} prompts')
print(f'KL sweep result:   {\"present\" if report.kl_sweep_result is not None else \"None\"}')
print(f'Attention results: {len(report.attention_results)} prompts (skipped)')
print(f'Output dir:        ${OUTPUT_DIR}')
"

echo ""
echo -e "${GREEN}=== KL Divergence Analysis Complete ===${NC}"
echo -e "Output directory: ${BLUE}${OUTPUT_DIR}/${CONCEPT}/${SAFE_MODEL}${NC}"
echo -e "Report:           ${BLUE}${OUTPUT_DIR}/${CONCEPT}/${SAFE_MODEL}/analysis_report.md${NC}"
echo -e "KL plots:         ${BLUE}${OUTPUT_DIR}/${CONCEPT}/${SAFE_MODEL}/${NC}"
echo -e "  - kl_prefix_vs_no_steer.pdf"
echo -e "  - kl_prefix_vs_all_steer.pdf"
echo -e "  - kl_prefix_length_sweep.pdf"
