#!/usr/bin/env bash
# =============================================================================
# run_steering.sh - Apply Steering Vectors to Model Generation
# =============================================================================
# Applies extracted steering vectors to model generation, steering model
# behavior towards or away from specific concepts.
#
# Usage:
#   ./scripts/apply_steering/run_steering.sh \
#       --vector data/vectors/refusal_Qwen_Qwen3-1.7B_mean.pt \
#       --model Qwen/Qwen3-1.7B
#
#   ./scripts/apply_steering/run_steering.sh \
#       --vector data/vectors/sentiment.pt \
#       --model Qwen/Qwen3-1.7B \
#       --samples 20 \
#       --multipliers "0.1,1.0,10.0" \
#       --evaluate
#
# Output:
#   data/steered/{concept}/{model}/layer{N}.jsonl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Apply steering vectors to model generation, steering behavior towards
or away from specific concepts.

Required Options:
    --vector PATH          Path to steering vector file (.pt)
    --model NAME           HuggingFace model name

Optional Options:
    --output DIR           Output directory (default: data/steered/)
    --samples N            Number of negative samples (default: 10)
    --multipliers LIST     Comma-separated multipliers (default: 0.01,0.1,1.0,10.0)
    --max-new-tokens N     Maximum tokens to generate (default: 100)
    --temperature F        Sampling temperature (default: 0.0 for greedy)
    --evaluate             Run evaluation on steered outputs
    --judge-model NAME     Judge model for LLM-as-judge evaluation
                           (default: google/gemini-3.1-flash-lite-preview)
    --mmlu-questions N     Number of MMLU questions for evaluation (default: 10)
    -h, --help             Show this help

Examples:
    # Basic steering
    $(basename "$0") \\
        --vector data/vectors/refusal_Qwen_Qwen3-1.7B_mean.pt \\
        --model Qwen/Qwen3-1.7B

    # Custom samples and multipliers
    $(basename "$0") \\
        --vector data/vectors/sentiment.pt \\
        --model Qwen/Qwen3-1.7B \\
        --samples 20 \\
        --multipliers "0.1,1.0,10.0"

    # With evaluation
    $(basename "$0") \\
        --vector data/vectors/sentiment.pt \\
        --model Qwen/Qwen3-1.7B \\
        --evaluate \\
        --mmlu-questions 20

Output Files:
    data/steered/{concept}/{model}/layer{N}.jsonl
    data/steered/{concept}/{model}/eval/layer{N}_mult{M}.json  (if --evaluate)

EOF
    exit 0
}

# Default values
VECTOR=""
MODEL=""
OUTPUT_DIR="$PROJECT_ROOT/data/steered/"
NUM_SAMPLES=10
MULTIPLIERS="0.01,0.1,1.0,10.0"
MAX_NEW_TOKENS=100
TEMPERATURE=0.0
EVALUATE=false
JUDGE_MODEL="google/gemini-3.1-flash-lite-preview"
MMLU_QUESTIONS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --vector)
            VECTOR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --multipliers)
            MULTIPLIERS="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --evaluate)
            EVALUATE=true
            shift
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --mmlu-questions)
            MMLU_QUESTIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VECTOR" ]]; then
    echo -e "${RED}Error: --vector is required${NC}"
    usage
fi

if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}"
    usage
fi

# Display configuration
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Steering Vector Application${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Vector:      ${GREEN}$VECTOR${NC}"
echo -e "Model:       ${GREEN}$MODEL${NC}"
echo -e "Output:      ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Samples:     ${GREEN}$NUM_SAMPLES${NC}"
echo -e "Multipliers: ${GREEN}$MULTIPLIERS${NC}"
echo -e "Max Tokens:  ${GREEN}$MAX_NEW_TOKENS${NC}"
echo -e "Temperature: ${GREEN}$TEMPERATURE${NC}"
if [[ "$EVALUATE" == true ]]; then
    echo -e "Evaluate:    ${GREEN}enabled${NC}"
    echo -e "Judge Model: ${GREEN}$JUDGE_MODEL${NC}"
    echo -e "MMLU Qs:     ${GREEN}$MMLU_QUESTIONS${NC}"
fi
echo -e "${BLUE}============================================${NC}"
echo ""

# Build command
CMD="uv run python -m steering_geometry.apply_steering \
    --vector \"$VECTOR\" \
    --model \"$MODEL\" \
    --output \"$OUTPUT_DIR\" \
    --samples $NUM_SAMPLES \
    --multipliers \"$MULTIPLIERS\" \
    --max-new-tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE"

if [[ "$EVALUATE" == true ]]; then
    CMD="$CMD --evaluate --judge-model \"$JUDGE_MODEL\" --mmlu-questions $MMLU_QUESTIONS"
fi

# Run steering
echo -e "${GREEN}Applying steering vector...${NC}"
eval "$CMD"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Steering application complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Results saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "${GREEN}============================================${NC}"
