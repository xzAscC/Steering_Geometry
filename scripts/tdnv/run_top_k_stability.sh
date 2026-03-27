#!/usr/bin/env bash
# =============================================================================
# run_top_k_stability.sh - TDNV Stability Analysis with Top-K Discriminative Tokens
# =============================================================================
# Analyzes how TDNV changes when using only top-k discriminative tokens per class.
#
# Scoring formula: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
# Higher score = token is closer to own class, farther from other class.
#
# Usage:
#   ./scripts/tdnv/run_top_k_stability.sh                                    # Default
#   ./scripts/tdnv/run_top_k_stability.sh -c honesty -k 16,32,64,128        # Custom k values
#   ./scripts/tdnv/run_top_k_stability.sh -c sentiment -m Qwen/Qwen3.5-2B   # Custom model
#
# Output:
#   - JSON: data/tdnv/top_k_stability/{concept}_{model}_k{k}.json
#   - Plot: plot/tdnv/top_k_stability/{concept}_{model}_stability.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

eval $(uv run python -m steering_geometry --shell)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Analyze TDNV stability when selecting top-k discriminative tokens.

TDNV measures separability between positive and negative activations.
This script varies the number of top-k discriminative tokens to analyze
how the selection affects TDNV stability across layers.

Options:
    -c, --concept CONCEPT    Concept to analyze (default: honesty)
                             Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --model MODEL        HuggingFace model name (default: Qwen/Qwen3-1.7B)
    -k, --k-values LIST      Comma-separated k values (default: 16,32,64,128,256)
    -p, --pairs N            Number of contrast pairs (default: 500)
    -o, --output DIR         Output directory for JSON results (default: data/tdnv/top_k_stability)
    --plot-dir DIR           Output directory for plots (default: plot/tdnv/top_k_stability)
    -l, --list               List available concepts
    -h, --help               Show this help

Examples:
    $(basename "$0")                                    # Default settings
    $(basename "$0") -c honesty -k 16,32,64,128        # Specific k values
    $(basename "$0") -c sentiment -m Qwen/Qwen3.5-2B   # Different model
    $(basename "$0") -c toxicity -k 8,16,32,64,128,256 # Extended k range

Output Files:
    JSON:  data/tdnv/top_k_stability/{concept}_{model}_k{k}.json
    Plot:  plot/tdnv/top_k_stability/{concept}_{model}_stability.pdf

EOF
    exit 0
}

list_available() {
    echo "Available concepts:"
    printf '  - %s\n' "${ALL_CONCEPTS[@]}"
    exit 0
}

CONCEPT="honesty"
MODEL="Qwen/Qwen3-1.7B"
K_VALUES="16,32,64,128,256"
NUM_PAIRS=500
OUTPUT_DIR="$PROJECT_ROOT/data/tdnv/top_k_stability"
PLOT_DIR="$PROJECT_ROOT/plot/tdnv/top_k_stability"

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
        -k|--k-values)
            K_VALUES="$2"
            shift 2
            ;;
        -p|--pairs)
            NUM_PAIRS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plot-dir)
            PLOT_DIR="$2"
            shift 2
            ;;
        -l|--list)
            list_available
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

IFS=',' read -ra K_ARRAY <<< "$K_VALUES"

mkdir -p "$OUTPUT_DIR" "$PLOT_DIR"

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV Top-K Stability Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:    ${GREEN}${CONCEPT}${NC}"
echo -e "Model:      ${GREEN}${MODEL}${NC}"
echo -e "K values:   ${GREEN}${K_ARRAY[*]}${NC}"
echo -e "Pairs:      ${GREEN}${NUM_PAIRS}${NC}"
echo -e "Output:     ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Plots:      ${YELLOW}${PLOT_DIR}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

RESULTS_LIST=()
K_LIST=()

for k in "${K_ARRAY[@]}"; do
    echo -e "${GREEN}Computing TDNV with top-${k} discriminative tokens...${NC}"
    echo "----------------------------------------"
    
    OUTPUT_FILE="${OUTPUT_DIR}/${CONCEPT}_${MODEL_SLUG}_k${k}.json"
    
    uv run python -m "steering_geometry.tdnv" \
        --concept "$CONCEPT" \
        --model "$MODEL" \
        --num-pairs "$NUM_PAIRS" \
        --top-k "$k" \
        --output "$OUTPUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done
    
    RESULTS_LIST+=("$OUTPUT_FILE")
    K_LIST+=("$k")
    
    echo ""
done

echo -e "${BLUE}Generating stability trend plot...${NC}"

PLOT_SCRIPT=$(cat << 'PYTHON_EOF'
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steering_geometry.tdnv import plot_stability_trend
from steering_geometry.types import TDNVResult

results_files = sys.argv[1].split(",")
k_values = [int(k) for k in sys.argv[2].split(",")]
output_path = sys.argv[3]
concept = sys.argv[4]
model = sys.argv[5]

results = []
for rf in results_files:
    with open(rf) as f:
        data = json.load(f)
    result = TDNVResult(
        concept=data["concept"],
        model_name=data["model_name"],
        num_pairs=data["num_pairs"],
        layers=data["layers"],
        tdnv_values=data["tdnv_values"],
        norm_num_values=data["norm_num_values"],
        norm_den_values=data["norm_den_values"],
        layerwise_energy=data["layerwise_energy"],
    )
    results.append(result)

plot_stability_trend(results, "Top-K", k_values, Path(output_path))
print(f"Plot saved to {output_path}")
PYTHON_EOF
)

uv run python -c "$PLOT_SCRIPT" \
    "$(IFS=','; echo "${RESULTS_LIST[*]}")" \
    "$(IFS=','; echo "${K_LIST[*]}")" \
    "${PLOT_DIR}/${CONCEPT}_${MODEL_SLUG}_stability.pdf" \
    "$CONCEPT" \
    "$MODEL"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ TDNV top-k stability analysis complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  JSON results: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Plots:        ${YELLOW}${PLOT_DIR}${NC}"
echo -e "${GREEN}============================================${NC}"
