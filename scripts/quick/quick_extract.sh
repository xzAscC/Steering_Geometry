#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat << EOF
Quick Extract - Single layer steering vector extraction

Usage: $(basename "$0") [OPTIONS]

Options:
    -c, --concept NAME   Concept to extract (default: honesty)
                         Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --model MODEL    HuggingFace model name (default: Qwen/Qwen3.5-2B)
    -l, --layer FLOAT    Relative layer position 0.0-1.0 (default: 0.5)
    -p, --pairs N        Number of contrast pairs (default: 500)
    -M, --method METHOD  Extraction method (default: mean)
                         Available: mean, pca, weighted_mean, discriminative
    -o, --output DIR     Output directory (default: data/vectors)
    -h, --help           Show this help

Examples:
    $(basename "$0")                           # Default: honesty, Qwen3.5-2B, layer 0.5
    $(basename "$0") -c toxicity -l 0.7        # Toxicity at layer 0.7
    $(basename "$0") -c sentiment -M pca       # Sentiment with PCA method
    $(basename "$0") -m google/gemma-2-2b      # Use Gemma model

EOF
    exit 0
}

CONCEPT="honesty"
MODEL="Qwen/Qwen3.5-2B"
LAYER="0.5"
NUM_PAIRS=500
METHOD="mean"
OUTPUT_DIR="$PROJECT_ROOT/data/vectors"

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
        -l|--layer)
            LAYER="$2"
            shift 2
            ;;
        -p|--pairs)
            NUM_PAIRS="$2"
            shift 2
            ;;
        -M|--method)
            METHOD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
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

mkdir -p "$OUTPUT_DIR"

safe_model="${MODEL//\//_}"

echo "============================================"
echo "Quick Extract - Single Layer"
echo "============================================"
echo "Concept:  $CONCEPT"
echo "Model:    $MODEL"
echo "Layer:    $LAYER"
echo "Pairs:    $NUM_PAIRS"
echo "Method:   $METHOD"
echo "Output:   $OUTPUT_DIR"
echo "============================================"
echo ""

uv run python -m steering_geometry.extract \
    --concept "$CONCEPT" \
    --model "$MODEL" \
    --num-pairs "$NUM_PAIRS" \
    --method "$METHOD" \
    --output "$OUTPUT_DIR" \
    --layers "$LAYER"

echo ""
echo "============================================"
echo "Output file:"
echo "  $OUTPUT_DIR/${CONCEPT}_${safe_model}_${METHOD}.pt"
echo "============================================"
