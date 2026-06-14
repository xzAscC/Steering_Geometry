#!/bin/bash
# =============================================================================
# steering_strength_prefix_sweep.sh - Steering Strength vs Prefix Length Sweep
# =============================================================================
# Runs a sweep evaluation over steering multipliers and prefix token counts,
# evaluating both concept effectiveness and MMLU-Pro capability retention.
#
# Usage:
#   ./scripts/experiments/steering_strength_prefix_sweep.sh [OPTIONS]
#
# Options:
#   -c, --concept              Concept to evaluate (default: sentiment)
#   -m, --model                Model name (default: Qwen/Qwen3-1.7B)
#   -v, --vector               Path to steering vector file
#                              (default: outputs/vectors/sentiment/discriminative/k128_layer0.7.pt)
#   -l, --layer                Layer fraction (default: 0.7)
#   --multipliers              Comma-separated list of multipliers
#                              (default: "0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0")
#   --steer-tokens             Comma-separated list of steer token counts
#                              (default: "1,2,3,5,10,20,50")
#   --include-full             Prepend None (all tokens) to steer_tokens: true/false
#                              (default: true)
#   -n, --num-samples          Number of prompt samples (default: 10)
#   --mmlu-pro-num-questions   Number of MMLU-Pro questions (default: 100)
#   --no-eval-concept          Disable concept evaluation
#   --no-eval-mmlu             Disable MMLU-Pro evaluation
#   --judge-model              Judge model for LLM-as-judge (default: google/gemini-3.1-flash-lite-preview)
#   --judge-api-base           Judge API base URL (default: https://openrouter.ai/api/v1)
#   -o, --output               Output directory (default: outputs/sweep_evaluation)
#   --log-level                Logging level (default: INFO)
#   -h, --help                 Show this help message
#
# Example:
#   ./scripts/experiments/steering_strength_prefix_sweep.sh -c sentiment -m "Qwen/Qwen3-1.7B"
#   ./scripts/experiments/steering_strength_prefix_sweep.sh --multipliers "0.1,1.0,5.0" --no-eval-mmlu
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
CONCEPT="sentiment"
MODEL="Qwen/Qwen3-1.7B"
VECTOR="$PROJECT_ROOT/outputs/vectors/sentiment/discriminative/k128_layer0.7.pt"
LAYER="0.7"
MULTIPLIERS="0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0"
STEER_TOKENS="1,2,3,5,10,20,50"
INCLUDE_FULL="true"
NUM_SAMPLES=10
MMLU_PRO_NUM_QUESTIONS=100
EVAL_CONCEPT="true"
EVAL_MMLU="true"
JUDGE_MODEL="google/gemini-3.1-flash-lite-preview"
JUDGE_API_BASE="https://openrouter.ai/api/v1"
OUTPUT_DIR="$PROJECT_ROOT/outputs/sweep_evaluation"
LOG_LEVEL="INFO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--concept)
            CONCEPT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -v|--vector)
            VECTOR="$2"
            shift 2
            ;;
        -l|--layer)
            LAYER="$2"
            shift 2
            ;;
        --multipliers)
            MULTIPLIERS="$2"
            shift 2
            ;;
        --steer-tokens)
            STEER_TOKENS="$2"
            shift 2
            ;;
        --include-full)
            INCLUDE_FULL="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --mmlu-pro-num-questions)
            MMLU_PRO_NUM_QUESTIONS="$2"
            shift 2
            ;;
        --no-eval-concept)
            EVAL_CONCEPT="false"
            shift
            ;;
        --no-eval-mmlu)
            EVAL_MMLU="false"
            shift
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --judge-api-base)
            JUDGE_API_BASE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --concept              Concept to evaluate (default: sentiment)"
            echo "  -m, --model                Model name (default: Qwen/Qwen3-1.7B)"
            echo "  -v, --vector               Path to steering vector file"
            echo "  -l, --layer                Layer fraction (default: 0.7)"
            echo "  --multipliers              Comma-separated multipliers"
            echo "                             (default: \"0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0\")"
            echo "  --steer-tokens             Comma-separated token counts"
            echo "                             (default: \"1,2,3,5,10,20,50\")"
            echo "  --include-full             Prepend None to steer_tokens: true/false (default: true)"
            echo "  -n, --num-samples          Number of prompt samples (default: 10)"
            echo "  --mmlu-pro-num-questions   Number of MMLU-Pro questions (default: 100)"
            echo "  --no-eval-concept          Disable concept evaluation"
            echo "  --no-eval-mmlu             Disable MMLU-Pro evaluation"
            echo "  --judge-model              Judge model (default: google/gemini-3.1-flash-lite-preview)"
            echo "  --judge-api-base           Judge API base URL (default: https://openrouter.ai/api/v1)"
            echo "  -o, --output               Output directory (default: outputs/sweep_evaluation)"
            echo "  --log-level                Logging level (default: INFO)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -c sentiment -m \"Qwen/Qwen3-1.7B\""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to Python list format
multipliers_python="[${MULTIPLIERS}]"

# Build steer_tokens Python list: optionally prepend None
steer_tokens_inner="${STEER_TOKENS}"
if [[ "$INCLUDE_FULL" == "true" ]]; then
    steer_tokens_python="[None, ${steer_tokens_inner}]"
else
    steer_tokens_python="[${steer_tokens_inner}]"
fi

# Convert booleans to Python
eval_concept_python="True"
eval_mmlu_python="True"
if [[ "$EVAL_CONCEPT" == "false" ]]; then
    eval_concept_python="False"
fi
if [[ "$EVAL_MMLU" == "false" ]]; then
    eval_mmlu_python="False"
fi

# Print header
echo ""
echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}Steering Strength x Prefix Length Sweep${NC}"
echo -e "${YELLOW}============================================${NC}"
echo -e "  Concept:      ${CYAN}${CONCEPT}${NC}"
echo -e "  Model:        ${CYAN}${MODEL}${NC}"
echo -e "  Vector:       ${CYAN}${VECTOR}${NC}"
echo -e "  Layer:        ${CYAN}${LAYER}${NC}"
echo -e "  Multipliers:  ${CYAN}${MULTIPLIERS}${NC}"
echo -e "  Steer tokens: ${CYAN}${STEER_TOKENS}${NC}"
echo -e "  Include full: ${CYAN}${INCLUDE_FULL}${NC}"
echo -e "  Num samples:  ${CYAN}${NUM_SAMPLES}${NC}"
echo -e "  MMLU-Pro:     ${CYAN}${MMLU_PRO_NUM_QUESTIONS} questions${NC}"
echo -e "  Eval concept: ${CYAN}${EVAL_CONCEPT}${NC}"
echo -e "  Eval MMLU:    ${CYAN}${EVAL_MMLU}${NC}"
echo -e "  Judge model:  ${CYAN}${JUDGE_MODEL}${NC}"
echo -e "  Output dir:   ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  Log level:    ${CYAN}${LOG_LEVEL}${NC}"
echo ""

# Run the sweep evaluation
uv run python -u -c "
from steering_geometry.sweep_evaluation import run_sweep_evaluation, plot_sweep_heatmaps
from steering_geometry.utils import configure_logging

configure_logging(level='${LOG_LEVEL}')

result = run_sweep_evaluation(
    concept='${CONCEPT}',
    model_name='${MODEL}',
    vector_path='${VECTOR}',
    layer_frac=${LAYER},
    multipliers=${multipliers_python},
    steer_tokens_values=${steer_tokens_python},
    num_samples=${NUM_SAMPLES},
    mmlu_pro_num_questions=${MMLU_PRO_NUM_QUESTIONS},
    evaluate_concept=${eval_concept_python},
    evaluate_mmlu=${eval_mmlu_python},
    judge_model='${JUDGE_MODEL}',
    judge_api_base='${JUDGE_API_BASE}',
    output_dir='${OUTPUT_DIR}',
)

paths = plot_sweep_heatmaps(result, output_dir='${OUTPUT_DIR}')
for p in paths:
    print(f'  Saved: {p}')
"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Sweep Evaluation Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
