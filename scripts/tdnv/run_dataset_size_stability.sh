#!/usr/bin/env bash
# =============================================================================
# run_dataset_size_stability.sh - TDNV Stability Across Dataset Sizes
# =============================================================================
# Analyzes how TDNV metrics vary across different contrast pair dataset sizes.
# Useful for determining the minimum dataset size needed for stable metrics.
#
# Usage:
#   ./scripts/tdnv/run_dataset_size_stability.sh
#   ./scripts/tdnv/run_dataset_size_stability.sh --concept polite --sizes 100,500,1000
#   ./scripts/tdnv/run_dataset_size_stability.sh -c sentiment -m Qwen/Qwen3-1.7B
#
# Output:
#   - JSON files: {output}/{concept}_{model}_{size}.json
#   - Trend plot: {output}/{concept}_{model}_trend.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Analyze TDNV stability across different dataset sizes.

Options:
    -c, --concept NAME     Concept to analyze (default: polite)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal, polite
    -m, --model NAME       HuggingFace model name (default: Qwen/Qwen3-1.7B)
    -s, --sizes LIST       Comma-separated list of dataset sizes (default: 100,500,1000,2000)
    -o, --output DIR       Output directory (default: outputs/tdnv/dataset_size/)
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # Default settings
    $(basename "$0") -c sentiment -s 100,500,1000       # Custom sizes
    $(basename "$0") -c honesty -m Qwen/Qwen3.5-2B      # Different model

Output Files:
    JSON:  {output}/{concept}_{model}_{size}.json
    Plot:  {output}/{concept}_{model}_trend.pdf

EOF
    exit 0
}

# Default parameters
CONCEPT="polite"
MODEL="Qwen/Qwen3-1.7B"
SIZES="100,500,1000,2000"
OUTPUT_DIR="$PROJECT_ROOT/outputs/tdnv/dataset_size/"

# Parse arguments
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
        -s|--sizes)
            SIZES="$2"
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
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert MODEL to safe filename (replace / with _)
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')

# Parse sizes into array
IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
TOTAL=${#SIZE_ARRAY[@]}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV Dataset Size Stability Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:    ${GREEN}${CONCEPT}${NC}"
echo -e "Model:      ${GREEN}${MODEL}${NC}"
echo -e "Sizes:      ${GREEN}${SIZES}${NC}"
echo -e "Output:     ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Total:      ${GREEN}${TOTAL} dataset size(s)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Process each size sequentially for reproducibility
CURRENT=0
for SIZE in "${SIZE_ARRAY[@]}"; do
    ((++CURRENT))
    echo -e "${GREEN}[$CURRENT/$TOTAL] Computing TDNV for size=$SIZE${NC}"
    echo "----------------------------------------"
    
    # Run TDNV computation
    uv run python -m steering_geometry.tdnv \
        --concept "$CONCEPT" \
        --model "$MODEL" \
        --num-pairs "$SIZE" \
        --output "$OUTPUT_DIR" \
        --plot-dir "$OUTPUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done
    
    # Rename output file to include size
    ORIGINAL_FILE="${OUTPUT_DIR}/${CONCEPT}_${MODEL_SLUG}.json"
    SIZED_FILE="${OUTPUT_DIR}/${CONCEPT}_${MODEL_SLUG}_${SIZE}.json"
    
    if [[ -f "$ORIGINAL_FILE" ]]; then
        mv "$ORIGINAL_FILE" "$SIZED_FILE"
        echo -e "  ${YELLOW}Renamed: ${ORIGINAL_FILE##*/} -> ${SIZED_FILE##*/}${NC}"
    else
        echo -e "  ${RED}Warning: Expected output file not found: ${ORIGINAL_FILE##*/}${NC}"
    fi
    
    echo ""
done

# Generate trend plot combining all results
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Generating trend plot...${NC}"
echo -e "${BLUE}============================================${NC}"

uv run python -c "
import json
from pathlib import Path

import matplotlib.pyplot as plt

output_dir = Path('${OUTPUT_DIR}')
model_slug = '${MODEL_SLUG}'
concept = '${CONCEPT}'

# Collect results
sizes = []
tdnv_values = []

for json_file in sorted(output_dir.glob(f'{concept}_{model_slug}_*.json')):
    with open(json_file) as f:
        data = json.load(f)
    
    # Extract size from filename
    size_str = json_file.stem.split('_')[-1]
    try:
        size = int(size_str)
        sizes.append(size)
        tdnv_values.append(data.get('overall_tdnv', 0.0))
    except ValueError:
        continue

if not sizes:
    print('No results found for plotting')
    exit(0)

# Sort by size
sorted_pairs = sorted(zip(sizes, tdnv_values))
sizes, tdnv_values = zip(*sorted_pairs)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sizes, tdnv_values, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Dataset Size (num pairs)', fontsize=12)
ax.set_ylabel('Overall TDNV', fontsize=12)
ax.set_title(f'TDNV Stability vs Dataset Size\nConcept: {concept}, Model: {model_slug}', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Add annotations for each point
for s, v in zip(sizes, tdnv_values):
    ax.annotate(f'{v:.4f}', (s, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()

# Save plot
plot_path = output_dir / f'{concept}_{model_slug}_trend.pdf'
plt.savefig(plot_path, format='pdf', dpi=150)
print(f'Saved trend plot: {plot_path}')
"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Dataset size stability analysis complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  JSON results: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Trend plot:   ${YELLOW}${OUTPUT_DIR}${CONCEPT}_${MODEL_SLUG}_trend.pdf${NC}"
echo -e "${GREEN}============================================${NC}"
