#!/usr/bin/env bash
# =============================================================================
# Prefix Steering Analysis — All Concepts
#
# Runs the attention analysis experiment for all three paper concepts:
#   refusal, polite, sentiment
#
# This tests whether the "prefix steering attention decay" pattern
# (prefix attention drops after steering stops, all-step maintains it)
# is consistent across different steering concepts.
#
# Usage:
#   ./run_all_concepts.sh [model] [layer_frac] [steer_tokens] [num_prompts] [max_new_tokens] [scale_mult]
#
# Examples:
#   ./run_all_concepts.sh                                    # defaults
#   ./run_all_concepts.sh Qwen/Qwen3-1.7B 0.7 10 10
#
# Output per concept:
#   outputs/prefix_analysis/{concept}/{model}/
#     ├── attention_to_prefix.pdf
#     ├── attention_cosine_shift.pdf
#     ├── kl_prefix_length_sweep.pdf
#     └── analysis_report.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONCEPTS="refusal polite sentiment"

MODEL="${1:-Qwen/Qwen3-1.7B}"
LAYER_FRAC="${2:-0.7}"
STEER_TOKENS="${3:-10}"
NUM_PROMPTS="${4:-10}"
MAX_NEW_TOKENS="${5:-100}"
SCALE_MULT="${6:-1.0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${BLUE}=== Prefix Steering Analysis: All Concepts ===${NC}"
echo -e "  Model:        ${GREEN}${MODEL}${NC}"
echo -e "  Layer:        ${GREEN}${LAYER_FRAC}${NC}"
echo -e "  Steer tokens: ${GREEN}${STEER_TOKENS}${NC}"
echo -e "  Concepts:     ${GREEN}${CONCEPTS}${NC}"
echo ""

cd "$PROJECT_ROOT"

FAILED=()
for CONCEPT in $CONCEPTS; do
    echo ""
    echo -e "${YELLOW}=== Running analysis for concept: ${CONCEPT} ===${NC}"
    echo ""

    if bash "${SCRIPT_DIR}/run_analysis.sh" \
        "$CONCEPT" "$MODEL" "$LAYER_FRAC" "$STEER_TOKENS" "$NUM_PROMPTS" "$MAX_NEW_TOKENS" "$SCALE_MULT"; then
        echo -e "${GREEN}✓ ${CONCEPT} complete${NC}"
    else
        echo -e "${RED}✗ ${CONCEPT} failed${NC}"
        FAILED+=("$CONCEPT")
    fi
done

echo ""
echo -e "${BLUE}=== Summary ===${NC}"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}All concepts completed successfully.${NC}"
else
    echo -e "${RED}Failed concepts: ${FAILED[*]}${NC}"
    exit 1
fi

echo ""
echo -e "Output directories:"
for CONCEPT in $CONCEPTS; do
    SAFE_MODEL="${MODEL//\//_}"
    DIR="outputs/prefix_analysis/${CONCEPT}/${SAFE_MODEL}"
    echo -e "  ${BLUE}${DIR}/${NC}"
done
