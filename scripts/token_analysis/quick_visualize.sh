#!/usr/bin/env bash
# =============================================================================
# run_visualize.sh - Visualize Discriminative Tokens
# =============================================================================
# Visualizes top discriminative tokens for a given concept across all layers.
#
# Usage:
#   ./scripts/token_analysis/run_visualize.sh --concept honesty
#   ./scripts/token_analysis/run_visualize.sh --concept toxicity --top-k 100
#   ./scripts/token_analysis/run_visualize.sh --concept sentiment --model Qwen/Qwen3.5-2B
#
# Output:
#   outputs/token_analysis/{concept}_{model}.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values (match token_analysis.py defaults)
MODEL="Qwen/Qwen3-1.7B"
TOP_K=50
OUTPUT_DIR="$PROJECT_ROOT/outputs/token_analysis/"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Visualize top discriminative tokens for a concept across layers.

Options:
    --concept NAME     Concept to analyze (required)
                       Available: honesty, sentiment, toxicity, sycophancy, refusal
    --model NAME       HuggingFace model name (default: sshleifer/tiny-gpt2)
    --output DIR       Output directory (default: outputs/token_analysis/)
    --top-k N          Number of top tokens to visualize (default: 50)
    -h, --help         Show this help

Examples:
    $(basename "$0") --concept honesty
    $(basename "$0") --concept toxicity --top-k 100
    $(basename "$0") --concept sentiment --model Qwen/Qwen3.5-2B

Output Files:
    outputs/token_analysis/{concept}_{model}.json

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
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
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
if [[ -z "${CONCEPT:-}" ]]; then
    echo "Error: --concept is required"
    usage
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Token Visualization"
echo "============================================"
echo "Concept:   $CONCEPT"
echo "Model:     $MODEL"
echo "Top-k:     $TOP_K"
echo "Output:    $OUTPUT_DIR"
echo "============================================"

uv run python -m steering_geometry.token_analysis visualize \
    --concept "$CONCEPT" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --top-k "$TOP_K"
