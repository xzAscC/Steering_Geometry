#!/usr/bin/env bash
# =============================================================================
# run_probe.sh - Probe Token-Level Separability
# =============================================================================
# Probes token-level separability across layers using linear classifiers.
#
# Usage:
#   ./scripts/token_analysis/run_probe.sh --concept honesty
#   ./scripts/token_analysis/run_probe.sh --concept toxicity --tokens-per-class 5000
#   ./scripts/token_analysis/run_probe.sh --concept sentiment --model Qwen/Qwen3.5-2B
#
# Output:
#   outputs/token_analysis/{concept}_{model}_probe.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values (match token_analysis.py defaults)
MODEL="sshleifer/tiny-gpt2"
TOKENS_PER_CLASS=10000
OUTPUT_DIR="$PROJECT_ROOT/outputs/token_analysis/"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Probe token-level separability across layers using linear classifiers.

Options:
    --concept NAME         Concept to probe (required)
                           Available: honesty, sentiment, toxicity, sycophancy, refusal
    --model NAME           HuggingFace model name (default: sshleifer/tiny-gpt2)
    --output DIR           Output directory (default: outputs/token_analysis/)
    --tokens-per-class N   Number of tokens to sample per class (default: 10000)
    -h, --help             Show this help

Examples:
    $(basename "$0") --concept honesty
    $(basename "$0") --concept toxicity --tokens-per-class 5000
    $(basename "$0") --concept sentiment --model Qwen/Qwen3.5-2B

Output Files:
    outputs/token_analysis/{concept}_{model}_probe.json

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
        --tokens-per-class)
            TOKENS_PER_CLASS="$2"
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
echo "Token Probe Analysis"
echo "============================================"
echo "Concept:          $CONCEPT"
echo "Model:            $MODEL"
echo "Tokens/class:     $TOKENS_PER_CLASS"
echo "Output:           $OUTPUT_DIR"
echo "============================================"

uv run python -m steering_geometry.token_analysis probe \
    --concept "$CONCEPT" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --tokens-per-class "$TOKENS_PER_CLASS"
