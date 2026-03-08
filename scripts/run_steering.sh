#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ALL_CONCEPTS=("honesty" "sycophancy" "toxicity" "sentiment" "refusal")
ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "google/gemma-2-2b")

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Apply steering vectors to model generation.

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    -s, --samples N        Number of samples to steer (default: 10)
    -M, --multipliers LIST Comma-separated multipliers (default: 0.01,0.1,1.0,10.0)
    -i, --input DIR        Input vector directory (default: data/vectors)
    -o, --output DIR       Output directory (default: data/steered)
    --max-tokens N         Maximum tokens to generate (default: 100)
    --temperature F        Sampling temperature (default: 0.0)
    --seed N               Random seed for sample selection (default: 42)
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # All concepts, default model
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
    $(basename "$0") -c sentiment -s 20 -M 0.5,1.0,2.0  # Custom params
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
NUM_SAMPLES=10
MULTIPLIERS="0.01,0.1,1.0,10.0"
INPUT_DIR="$PROJECT_ROOT/data/vectors"
OUTPUT_DIR="$PROJECT_ROOT/data/steered"
MAX_TOKENS=100
TEMPERATURE=0.0
SEED=42

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
        -s|--samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -M|--multipliers)
            MULTIPLIERS="$2"
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
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
echo "Steering Vector Application Pipeline"
echo "============================================"
echo "Concepts:     ${CONCEPT_ARRAY[*]}"
echo "Models:       ${MODEL_ARRAY[*]}"
echo "Samples:      $NUM_SAMPLES"
echo "Multipliers:  $MULTIPLIERS"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Max tokens:   $MAX_TOKENS"
echo "Temperature:  $TEMPERATURE"
echo "Seed:         $SEED"
echo "Total:        $total steering run(s)"
echo "============================================"
echo ""

for model in "${MODEL_ARRAY[@]}"; do
    safe_model="${model//\//_}"
    for concept in "${CONCEPT_ARRAY[@]}"; do
        ((current++))
        vector_file="$INPUT_DIR/${concept}_${safe_model}_mean.pt"
        
        echo "[$current/$total] Steering: $concept × $model"
        echo "----------------------------------------"
        
        if [[ ! -f "$vector_file" ]]; then
            echo "  WARNING: Vector file not found: $vector_file"
            echo "  Run extraction first: ./scripts/run_extractions.sh -c $concept -m $model"
            echo ""
            continue
        fi
        
        uv run python -m "steering_geometry.apply_steering" \
            --vector "$vector_file" \
            --model "$model" \
            --output "$OUTPUT_DIR" \
            --samples "$NUM_SAMPLES" \
            --multipliers "$MULTIPLIERS" \
            --max-new-tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" 2>&1 | while read -r line; do
                echo "  $line"
            done
        
        echo ""
    done
done

echo "============================================"
echo "✓ All steering runs complete!"
echo "  Results saved to: $OUTPUT_DIR"
echo "============================================"
