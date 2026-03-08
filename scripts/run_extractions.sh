#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ALL_CONCEPTS=("honesty" "sycophancy" "toxicity" "sentiment" "refusal")
ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "google/gemma-2-2b")

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run steering vector extraction for specified concepts and models.

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    -p, --pairs N          Number of contrast pairs (default: 500)
    -M, --method METHOD    Extraction method: mean or pca (default: mean)
    -o, --output DIR       Output directory (default: data/vectors)
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts, default model
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
    $(basename "$0") -c sentiment -p 100 -M pca         # Custom params
    $(basename "$0") -c all -m all                      # All concepts × all models

EOF
    exit 0
}

list_available() {
    echo "Available concepts:"
    printf '  - %s\n' "${ALL_CONCEPTS[@]}"
    echo ""
    echo "Available models:"
    printf '  - %s\n' "${ALL_MODELS[@]}"
    exit 0
}

CONCEPTS=""
MODELS="Qwen/Qwen3.5-2B"
NUM_PAIRS=500
METHOD="mean"
OUTPUT_DIR="$PROJECT_ROOT/data/vectors"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concepts)
            CONCEPTS="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
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

IFS=',' read -ra CONCEPT_ARRAY <<< "$CONCEPTS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

if [[ "$CONCEPTS" == "all" || -z "$CONCEPTS" ]]; then
    CONCEPT_ARRAY=("${ALL_CONCEPTS[@]}")
fi

if [[ "$MODELS" == "all" ]]; then
    MODEL_ARRAY=("${ALL_MODELS[@]}")
fi

mkdir -p "$OUTPUT_DIR"

total=$((${#CONCEPT_ARRAY[@]} * ${#MODEL_ARRAY[@]}))
current=0

echo "============================================"
echo "Steering Vector Extraction Pipeline"
echo "============================================"
echo "Concepts: ${CONCEPT_ARRAY[*]}"
echo "Models:   ${MODEL_ARRAY[*]}"
echo "Pairs:    $NUM_PAIRS"
echo "Method:   $METHOD"
echo "Output:   $OUTPUT_DIR"
echo "Total:    $total extraction(s)"
echo "============================================"
echo ""

for model in "${MODEL_ARRAY[@]}"; do
    for concept in "${CONCEPT_ARRAY[@]}"; do
        ((current++))
        echo "[$current/$total] Extracting: $concept × $model"
        echo "----------------------------------------"
        
        uv run python -m "steering_geometry.extract" \
            --concept "$concept" \
            --model "$model" \
            --num-pairs "$NUM_PAIRS" \
            --method "$METHOD" \
            --output "$OUTPUT_DIR" 2>&1 | while read -r line; do
                echo "  $line"
            done
        
        echo ""
    done
done

echo "============================================"
echo "✓ All extractions complete!"
echo "  Vectors saved to: $OUTPUT_DIR"
echo "============================================"
