#!/usr/bin/env bash
# =============================================================================
# run_seed_stability.sh - TDNV Stability Analysis Across Random Seeds
# =============================================================================
# Analyzes how TDNV metrics vary when using different random seeds for
# contrast pair sampling. Demonstrates stability of TDNV as a metric.
#
# Usage:
#   ./scripts/tdnv/run_seed_stability.sh                           # Default params
#   ./scripts/tdnv/run_seed_stability.sh --concept honesty         # Specific concept
#   ./scripts/tdnv/run_seed_stability.sh --seeds 0,1,2,3,4         # Custom seeds
#   ./scripts/tdnv/run_seed_stability.sh --concept all --seeds all # Full sweep
#
# Output:
#   JSON:  outputs/tdnv/seed/{concept}_{model}_seed{N}.json
#   Plot:  outputs/tdnv/seed/{concept}_{model}_stability.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default parameters
CONCEPT="polite"
MODEL="Qwen/Qwen3-1.7B"
SEEDS="0,1,2,3,4"
NUM_PAIRS=500
OUTPUT_DIR="$PROJECT_ROOT/outputs/tdnv/seed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Analyze TDNV stability across different random seeds for contrast pair sampling.

TDNV should be stable (low variance) across seeds if it's a reliable metric.

Options:
    --concept NAME     Concept to analyze (default: polite)
                       Available: polite, sentiment, refusal, honesty, sycophancy, toxicity
    --model NAME       HuggingFace model name (default: Qwen/Qwen3-1.7B)
    --seeds LIST       Comma-separated list of seeds (default: 0,1,2,3,4)
                       Use "all" for extended sweep: 0,1,2,3,4,5,6,7,8,9
    --num-pairs N      Number of contrast pairs (default: 500)
    --output DIR       Output directory (default: outputs/tdnv/seed)
    -h, --help         Show this help

Examples:
    $(basename "$0")                              # Default: polite, 5 seeds
    $(basename "$0") --concept honesty            # Analyze honesty concept
    $(basename "$0") --seeds 0,1,2,3,4,5,6,7,8,9  # Extended 10-seed sweep
    $(basename "$0") --concept sentiment --num-pairs 1000

Output Files:
    JSON:  outputs/tdnv/seed/{concept}_{model}_seed{N}.json
    Plot:  outputs/tdnv/seed/{concept}_{model}_stability.pdf

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --num-pairs)
            NUM_PAIRS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Handle "all" seeds option
if [[ "$SEEDS" == "all" ]]; then
    SEEDS="0,1,2,3,4,5,6,7,8,9"
fi

# Parse seeds into array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

mkdir -p "$OUTPUT_DIR"

MODEL_SAFE="${MODEL//\//_}"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV Seed Stability Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:    ${GREEN}$CONCEPT${NC}"
echo -e "Model:      ${GREEN}$MODEL${NC}"
echo -e "Seeds:      ${GREEN}${SEED_ARRAY[*]}${NC}"
echo -e "Pairs:      ${GREEN}$NUM_PAIRS${NC}"
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Collect results for summary
declare -a RESULTS

total=${#SEED_ARRAY[@]}
current=0

for seed in "${SEED_ARRAY[@]}"; do
    ((++current))
    echo -e "${GREEN}[$current/$total] Running with seed=$seed${NC}"
    echo "----------------------------------------"
    
    export PYTHONHASHSEED="$seed"
    
    OUTPUT_FILE="$OUTPUT_DIR/${CONCEPT}_${MODEL_SAFE}_seed${seed}.json"
    
    cd "$PROJECT_ROOT"
    
    uv run python -m steering_geometry.tdnv \
        --concept "$CONCEPT" \
        --model "$MODEL" \
        --num-pairs "$NUM_PAIRS" \
        --output "$OUTPUT_DIR" \
        --plot-dir "$OUTPUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done
    
    DEFAULT_OUTPUT="$OUTPUT_DIR/${CONCEPT}_${MODEL_SAFE}.json"
    if [[ -f "$DEFAULT_OUTPUT" ]]; then
        mv "$DEFAULT_OUTPUT" "$OUTPUT_FILE"
        RESULTS+=("$seed:$OUTPUT_FILE")
        echo -e "  ${YELLOW}Saved: $OUTPUT_FILE${NC}"
    fi
    
    echo ""
done

# Summary
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Seed Stability Analysis Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Output files:"
for result in "${RESULTS[@]}"; do
    seed="${result%%:*}"
    file="${result#*:}"
    echo -e "  ${YELLOW}Seed $seed: $file${NC}"
done
echo ""
echo -e "To analyze stability, compare TDNV values across seeds."
echo -e "Low variance indicates stable TDNV metric."
echo -e "${GREEN}============================================${NC}"
