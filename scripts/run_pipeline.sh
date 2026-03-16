#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - Complete Steering Vector Pipeline
# =============================================================================
# Orchestrates the full pipeline: Extract → Steer → Evaluate
#
# Usage:
#   ./scripts/run_pipeline.sh                                    # Full pipeline, default model
#   ./scripts/run_pipeline.sh -c honesty,toxicity                # Specific concepts
#   ./scripts/run_pipeline.sh -m Qwen/Qwen3.5-2B,google/gemma-2-2b  # Multiple models
#   ./scripts/run_pipeline.sh --skip-extract                     # Skip extraction step
#   ./scripts/run_pipeline.sh --eval-only                        # Only run evaluation
#
# Pipeline Steps:
#   1. Extract steering vectors from contrast pairs
#   2. Apply steering vectors to generate steered outputs
#   3. Evaluate steering effectiveness (optional LLM-as-judge + MMLU)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Available options
ALL_CONCEPTS=("honesty" "sycophancy" "toxicity" "sentiment" "refusal")
ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "google/gemma-2-2b")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run the complete steering vector pipeline: Extract → Steer → Evaluate

Pipeline Steps:
  1. EXTRACT  - Extract steering vectors from contrast pairs
  2. STEER    - Apply steering vectors to generate steered outputs
  3. EVALUATE - Evaluate steering effectiveness (optional LLM-as-judge)

Options:
    -c, --concepts LIST    Comma-separated list of concepts (default: all)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal
    -m, --models LIST      Comma-separated list of models (default: Qwen/Qwen3.5-2B)
    
    # Extraction options
    -p, --pairs N          Number of contrast pairs (default: 500)
    -M, --method METHOD    Extraction method: mean or pca (default: mean)
    
    # Steering options
    -s, --samples N        Number of samples to steer (default: 10)
    --multipliers LIST     Comma-separated multipliers (default: 0.01,0.1,1.0,10.0)
    --max-tokens N         Maximum tokens to generate (default: 100)
    --temperature F        Sampling temperature (default: 0.0)
    
    # Evaluation options
    --evaluate             Run LLM-as-judge and MMLU evaluation
    --judge-model MODEL    Judge model (default: google/gemini-3.1-flash-lite-preview)
    --mmlu-questions N     Number of MMLU questions (default: 10)
    
    # Pipeline control
    --skip-extract         Skip extraction step (use existing vectors)
    --skip-steer           Skip steering step (use existing outputs)
    --eval-only            Only run evaluation (skip extract + steer)
    --extract-only         Only run extraction
    --steer-only           Only run steering (requires vectors)
    
    # Other
    -o, --output DIR       Base output directory (default: data)
    -l, --list             List available concepts and models
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # Full pipeline, all concepts
    $(basename "$0") -c honesty,toxicity                # Specific concepts
    $(basename "$0") -m Qwen/Qwen3.5-2B --evaluate      # With evaluation
    $(basename "$0") --skip-extract                     # Skip extraction
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

# Default values
CONCEPTS=""
MODELS="Qwen/Qwen3.5-2B"
NUM_PAIRS=500
METHOD="mean"
NUM_SAMPLES=10
MULTIPLIERS="0.01,0.1,1.0,10.0"
MAX_TOKENS=100
TEMPERATURE=0.0
BASE_OUTPUT="$PROJECT_ROOT/data"
EVALUATE=false
JUDGE_MODEL="google/gemini-3.1-flash-lite-preview"
MMLU_QUESTIONS=10

# Pipeline control flags
SKIP_EXTRACT=false
SKIP_STEER=false
EVAL_ONLY=false
EXTRACT_ONLY=false
STEER_ONLY=false

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
        -s|--samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --multipliers)
            MULTIPLIERS="$2"
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
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-steer)
            SKIP_STEER=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        --steer-only)
            STEER_ONLY=true
            shift
            ;;
        -o|--output)
            BASE_OUTPUT="$2"
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

# Parse concepts and models
IFS=',' read -ra CONCEPT_ARRAY <<< "$CONCEPTS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

if [[ "$CONCEPTS" == "all" || -z "$CONCEPTS" ]]; then
    CONCEPT_ARRAY=("${ALL_CONCEPTS[@]}")
fi

if [[ "$MODELS" == "all" ]]; then
    MODEL_ARRAY=("${ALL_MODELS[@]")
fi

# Set up output directories
VECTOR_DIR="$BASE_OUTPUT/vectors"
STEERED_DIR="$BASE_OUTPUT/steered"

mkdir -p "$VECTOR_DIR" "$STEERED_DIR"

# Calculate total runs
total=$((${#CONCEPT_ARRAY[@]} * ${#MODEL_ARRAY[@]}))

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Steering Vector Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concepts:     ${GREEN}${CONCEPT_ARRAY[*]}${NC}"
echo -e "Models:       ${GREEN}${MODEL_ARRAY[*]}${NC}"
echo -e "Total runs:   ${GREEN}$total${NC}"
echo ""
echo -e "Pipeline Configuration:"
echo -e "  Extract:    $([ "$SKIP_EXTRACT" == false ] && echo -e "${GREEN}YES${NC}" || echo -e "${YELLOW}SKIP${NC}")"
echo -e "  Steer:      $([ "$SKIP_STEER" == false ] && echo -e "${GREEN}YES${NC}" || echo -e "${YELLOW}SKIP${NC}")"
echo -e "  Evaluate:   $([ "$EVALUATE" == true ] && echo -e "${GREEN}YES${NC}" || echo -e "${YELLOW}NO${NC}")"
echo ""
echo -e "Parameters:"
echo -e "  Pairs:          $NUM_PAIRS"
echo -e "  Method:         $METHOD"
echo -e "  Samples:        $NUM_SAMPLES"
echo -e "  Multipliers:    $MULTIPLIERS"
echo -e "  Max tokens:     $MAX_TOKENS"
echo -e "  Temperature:    $TEMPERATURE"
echo -e "${BLUE}============================================${NC}"
echo ""

# =============================================================================
# STEP 1: EXTRACT
# =============================================================================
run_extract() {
    if [[ "$SKIP_EXTRACT" == true || "$EVAL_ONLY" == true ]]; then
        echo -e "${YELLOW}[SKIP] Extraction step${NC}"
        return
    fi
    
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}STEP 1: EXTRACTING STEERING VECTORS${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    for model in "${MODEL_ARRAY[@]}"; do
        for concept in "${CONCEPT_ARRAY[@]}"; do
            echo -e "\n${GREEN}Extracting: $concept × $model${NC}"
            echo "----------------------------------------"
            
            uv run python -m "steering_geometry.extract" \
                --concept "$concept" \
                --model "$model" \
                --num-pairs "$NUM_PAIRS" \
                --method "$METHOD" \
                --output "$VECTOR_DIR" 2>&1 | while read -r line; do
                    echo "  $line"
                done
        done
    done
    
    echo -e "\n${GREEN}✓ Extraction complete!${NC}"
    echo -e "  Vectors saved to: ${YELLOW}$VECTOR_DIR${NC}"
}

# =============================================================================
# STEP 2: STEER
# =============================================================================
run_steer() {
    if [[ "$SKIP_STEER" == true || "$EVAL_ONLY" == true ]]; then
        echo -e "${YELLOW}[SKIP] Steering step${NC}"
        return
    fi
    
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}STEP 2: APPLYING STEERING VECTORS${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    current=0
    for model in "${MODEL_ARRAY[@]}"; do
        safe_model="${model//\//_}"
        for concept in "${CONCEPT_ARRAY[@]}"; do
            ((current++))
            vector_file="$VECTOR_DIR/${concept}_${safe_model}_${METHOD}.pt"
            
            echo -e "\n${GREEN}[$current/$total] Steering: $concept × $model${NC}"
            echo "----------------------------------------"
            
            if [[ ! -f "$vector_file" ]]; then
                echo -e "  ${RED}WARNING: Vector file not found: $vector_file${NC}"
                echo -e "  ${YELLOW}Run extraction first or use --skip-steer${NC}"
                continue
            fi
            
            uv run python -m "steering_geometry.apply_steering" \
                --vector "$vector_file" \
                --model "$model" \
                --output "$STEERED_DIR" \
                --samples "$NUM_SAMPLES" \
                --multipliers "$MULTIPLIERS" \
                --max-new-tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" 2>&1 | while read -r line; do
                    echo "  $line"
                done
        done
    done
    
    echo -e "\n${GREEN}✓ Steering complete!${NC}"
    echo -e "  Results saved to: ${YELLOW}$STEERED_DIR${NC}"
}

# =============================================================================
# STEP 3: EVALUATE
# =============================================================================
run_evaluate() {
    if [[ "$EVALUATE" == false ]]; then
        echo -e "${YELLOW}[SKIP] Evaluation step (use --evaluate to enable)${NC}"
        return
    fi
    
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}STEP 3: EVALUATING STEERING EFFECTIVENESS${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    EVAL_FLAGS="--evaluate --judge-model $JUDGE_MODEL --mmlu-questions $MMLU_QUESTIONS"
    
    current=0
    for model in "${MODEL_ARRAY[@]}"; do
        safe_model="${model//\//_}"
        for concept in "${CONCEPT_ARRAY[@]}"; do
            ((current++))
            vector_file="$VECTOR_DIR/${concept}_${safe_model}_${METHOD}.pt"
            
            echo -e "\n${GREEN}[$current/$total] Evaluating: $concept × $model${NC}"
            echo "----------------------------------------"
            
            if [[ ! -f "$vector_file" ]]; then
                echo -e "  ${RED}WARNING: Vector file not found: $vector_file${NC}"
                continue
            fi
            
            uv run python -m "steering_geometry.apply_steering" \
                --vector "$vector_file" \
                --model "$model" \
                --output "$STEERED_DIR" \
                --samples "$NUM_SAMPLES" \
                --multipliers "$MULTIPLIERS" \
                $EVAL_FLAGS 2>&1 | while read -r line; do
                    echo "  $line"
                done
        done
    done
    
    echo -e "\n${GREEN}✓ Evaluation complete!${NC}"
    echo -e "  Results saved to: ${YELLOW}$STEERED_DIR${NC}"
}

# =============================================================================
# RUN PIPELINE
# =============================================================================

# Handle --extract-only and --steer-only flags
if [[ "$EXTRACT_ONLY" == true ]]; then
    run_extract
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}✓ Pipeline complete (Extract only)!${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
fi

if [[ "$STEER_ONLY" == true ]]; then
    run_steer
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}✓ Pipeline complete (Steer only)!${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
fi

# Run full pipeline
run_extract
run_steer
run_evaluate

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Vectors:  ${YELLOW}$VECTOR_DIR${NC}"
echo -e "  Outputs:  ${YELLOW}$STEERED_DIR${NC}"
echo -e "${GREEN}============================================${NC}"
