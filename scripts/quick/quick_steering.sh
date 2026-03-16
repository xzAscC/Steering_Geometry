#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat << EOF
Quick Steering - Apply steering vector (single layer)

Usage: $(basename "$0") [OPTIONS]

Options:
    -c, --concept NAME     Concept to steer (default: honesty)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --model MODEL      HuggingFace model name (default: Qwen/Qwen3.5-2B)
    -l, --layer FLOAT      Relative layer position (default: 0.5)
                           Note: Must match extraction layer
    -s, --samples N        Number of samples to steer (default: 10)
    -M, --multiplier F     Single multiplier value (default: 1.0)
    -i, --input DIR        Input vector directory (default: data/vectors)
    -o, --output DIR       Output directory (default: data/steered)
    --max-tokens N         Max tokens to generate (default: 100)
    -h, --help             Show this help

Prerequisites:
    Run quick_extract.sh first to generate the steering vector.

Examples:
    $(basename "$0")                           # Default: honesty, multiplier 1.0
    $(basename "$0") -c toxicity -M 2.0        # Toxicity with multiplier 2.0
    $(basename "$0") -c sentiment -s 5         # 5 samples only
    $(basename "$0") -l 0.7 -M 0.5             # Layer 0.7, multiplier 0.5

EOF
    exit 0
}

CONCEPT="honesty"
MODEL="Qwen/Qwen3.5-2B"
LAYER="0.5"
NUM_SAMPLES=10
MULTIPLIER="1.0"
INPUT_DIR="$PROJECT_ROOT/data/vectors"
OUTPUT_DIR="$PROJECT_ROOT/data/steered"
MAX_TOKENS=100

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
        -s|--samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -M|--multiplier)
            MULTIPLIER="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
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
VECTOR_FILE="$INPUT_DIR/${CONCEPT}_${safe_model}_mean.pt"

if [[ ! -f "$VECTOR_FILE" ]]; then
    echo "ERROR: Vector file not found: $VECTOR_FILE"
    echo ""
    echo "Run extraction first:"
    echo "  ./scripts/quick/quick_extract.sh -c $CONCEPT -m $MODEL -l $LAYER"
    exit 1
fi

echo "============================================"
echo "Quick Steering - Single Layer"
echo "============================================"
echo "Concept:     $CONCEPT"
echo "Model:       $MODEL"
echo "Layer:       $LAYER"
echo "Samples:     $NUM_SAMPLES"
echo "Multiplier:  $MULTIPLIER"
echo "Vector:      $VECTOR_FILE"
echo "Output:      $OUTPUT_DIR"
echo "Max tokens:  $MAX_TOKENS"
echo "============================================"
echo ""

uv run python -m steering_geometry.apply_steering \
    --vector "$VECTOR_FILE" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --samples "$NUM_SAMPLES" \
    --multipliers "$MULTIPLIER" \
    --max-new-tokens "$MAX_TOKENS"

echo ""
echo "============================================"
echo "Steering complete!"
echo "Results saved to: $OUTPUT_DIR/$CONCEPT/$safe_model/"
echo "============================================"
