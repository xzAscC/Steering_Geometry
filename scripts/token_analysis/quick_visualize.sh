#!/usr/bin/env bash
# =============================================================================
# quick_visualize.sh - Visualize Discriminative Tokens
# =============================================================================
# Visualizes top discriminative tokens for one or more concepts across layers.
#
# Usage:
#   ./scripts/token_analysis/quick_visualize.sh --concepts polite
#   ./scripts/token_analysis/quick_visualize.sh --concepts polite,sentiment,refusal
#   ./scripts/token_analysis/quick_visualize.sh --concepts refusal --top-k 100
#   ./scripts/token_analysis/quick_visualize.sh --concepts sentiment --model Qwen/Qwen3.5-2B
#
# Output:
#   outputs/token_analysis/{concept}_{model}.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
MODEL="Qwen/Qwen3-1.7B"
CONCEPTS="polite"
TOP_K=50
LAST_N=""
OUTPUT_DIR="$PROJECT_ROOT/outputs/token_analysis/"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Visualize top discriminative tokens for one or more concepts across layers.

Options:
    --concepts LIST    Comma-separated concepts to analyze (required)
                       Available: polite, sentiment, refusal
    --model NAME       HuggingFace model name (default: Qwen/Qwen3-1.7B)
    --output DIR       Output directory (default: outputs/token_analysis/)
    --top-k N          Number of top tokens to visualize (default: 50)
    --last-n N         Only use the last N tokens per sequence for scoring
    -h, --help         Show this help

Examples:
    $(basename "$0") --concepts polite
    $(basename "$0") --concepts polite,sentiment,refusal
    $(basename "$0") --concepts refusal --top-k 100
    $(basename "$0") --concepts sentiment --model Qwen/Qwen3.5-2B
    $(basename "$0") --concepts polite --last-n 10

Output Files:
    outputs/token_analysis/{concept}_{model}.json

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concepts)
            CONCEPTS="$2"
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
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --last-n)
            LAST_N="$2"
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

# Check required arguments
if [[ -z "${CONCEPTS:-}" ]]; then
    echo "Error: --concepts is required"
    usage
fi

mkdir -p "$OUTPUT_DIR"

# Split comma-separated concepts into array
IFS="," read -ra CONCEPT_LIST <<< "$CONCEPTS"

echo "============================================"
echo "Token Visualization"
echo "============================================"
echo "Concepts:  ${CONCEPT_LIST[*]}"
echo "Model:     $MODEL"
echo "Top-k:     $TOP_K"
echo "Last-n:    ${LAST_N:-all tokens}"
echo "Output:    $OUTPUT_DIR"
echo "============================================"

for CONCEPT in "${CONCEPT_LIST[@]}"; do
    echo ""
    echo "--- Visualizing: $CONCEPT ---"
    uv run python -m steering_geometry.token_analysis visualize \
        --concept "$CONCEPT" \
        --model "$MODEL" \
        --output "$OUTPUT_DIR" \
        --top-k "$TOP_K" \
        ${LAST_N:+--last-n "$LAST_N"}
done
