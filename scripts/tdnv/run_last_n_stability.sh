#!/usr/bin/env bash
# =============================================================================
# run_last_n_stability.sh - TDNV Stability Analysis for Last-N Token Selection
# =============================================================================
# Analyzes how TDNV metrics vary when selecting different numbers of last tokens.
# Compares TDNV when using only last token (common in extraction) vs more tokens.
#
# Usage:
#   ./scripts/tdnv/run_last_n_stability.sh
#   ./scripts/tdnv/run_last_n_stability.sh --concept polite --n-values 1,5,10,20
#   ./scripts/tdnv/run_last_n_stability.sh -c sentiment -m Qwen/Qwen3-1.7B
#
# Output:
#   - JSON files: {output}/{concept}_{model}_last{n}.json
#   - Trend plot: {output}/{concept}_{model}_last_n_trend.pdf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Analyze TDNV stability across different last-n token selections.
Lower n = fewer tokens (more selective), higher n = more tokens (more inclusive).

Options:
    -c, --concept NAME     Concept to analyze (default: polite)
                           Available: honesty, sycophancy, toxicity, sentiment, refusal, polite
    -m, --model NAME       HuggingFace model name (default: Qwen/Qwen3-1.7B)
    -n, --n-values LIST    Comma-separated list of last-n values (default: 1,5,10,20,50)
    -p, --pairs N          Number of contrast pairs (default: 500)
    -o, --output DIR       Output directory (default: outputs/tdnv/last_n/)
    -h, --help             Show this help

Examples:
    $(basename "$0")                                    # Default settings
    $(basename "$0") -c sentiment -n 1,3,5,10           # Custom n values
    $(basename "$0") -c honesty -m Qwen/Qwen3.5-2B      # Different model

Output Files:
    JSON:  {output}/{concept}_{model}_last{n}.json
    Plot:  {output}/{concept}_{model}_last_n_trend.pdf

EOF
    exit 0
}

CONCEPT="polite"
MODEL="Qwen/Qwen3-1.7B"
N_VALUES="1,5,10,20,50"
NUM_PAIRS=500
OUTPUT_DIR="$PROJECT_ROOT/outputs/tdnv/last_n/"

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
        -n|--n-values)
            N_VALUES="$2"
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
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')

IFS=',' read -ra N_ARRAY <<< "$N_VALUES"
TOTAL=${#N_ARRAY[@]}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}TDNV Last-N Token Stability Analysis${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Concept:    ${GREEN}${CONCEPT}${NC}"
echo -e "Model:      ${GREEN}${MODEL}${NC}"
echo -e "N values:   ${GREEN}${N_VALUES}${NC}"
echo -e "Pairs:      ${GREEN}${NUM_PAIRS}${NC}"
echo -e "Output:     ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "Total:      ${GREEN}${TOTAL} n-value(s)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

CURRENT=0
for N_VAL in "${N_ARRAY[@]}"; do
    ((++CURRENT))
    echo -e "${GREEN}[$CURRENT/$TOTAL] Computing TDNV for last_n=$N_VAL${NC}"
    echo "----------------------------------------"
    
    uv run python -m steering_geometry.tdnv \
        --concept "$CONCEPT" \
        --model "$MODEL" \
        --num-pairs "$NUM_PAIRS" \
        --last-n "$N_VAL" \
        --output "$OUTPUT_DIR" \
        --plot-dir "$OUTPUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done
    
    ORIGINAL_FILE="${OUTPUT_DIR}/${CONCEPT}_${MODEL_SLUG}.json"
    NAMED_FILE="${OUTPUT_DIR}/${CONCEPT}_${MODEL_SLUG}_last${N_VAL}.json"
    
    if [[ -f "$ORIGINAL_FILE" ]]; then
        mv "$ORIGINAL_FILE" "$NAMED_FILE"
        echo -e "  ${YELLOW}Renamed: ${ORIGINAL_FILE##*/} -> ${NAMED_FILE##*/}${NC}"
    else
        echo -e "  ${RED}Warning: Expected output file not found: ${ORIGINAL_FILE##*/}${NC}"
    fi
    
    echo ""
done

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

n_values = []
tdnv_by_layer = {}

for json_file in sorted(output_dir.glob(f'{concept}_{model_slug}_last*.json')):
    with open(json_file) as f:
        data = json.load(f)
    
    n_str = json_file.stem.split('_last')[-1]
    try:
        n_val = int(n_str)
        n_values.append(n_val)
        
        for layer_idx, layer in enumerate(data.get('layers', [])):
            if layer not in tdnv_by_layer:
                tdnv_by_layer[layer] = {}
            tdnv_by_layer[layer][n_val] = data.get('tdnv_values', [0])[layer_idx]
    except ValueError:
        continue

if not n_values:
    print('No results found for plotting')
    exit(0)

n_values = sorted(n_values)
layers = sorted(tdnv_by_layer.keys())
num_layers = len(layers)
cmap = plt.get_cmap('viridis')
colors = [cmap(i / max(num_layers - 1, 1)) for i in range(num_layers)]

fig, ax = plt.subplots(figsize=(12, 6))

for layer_idx, layer in enumerate(layers):
    tdnv_vals = [tdnv_by_layer[layer].get(n, float('nan')) for n in n_values]
    ax.plot(n_values, tdnv_vals, '-o', color=colors[layer_idx], 
            label=f'Layer {layer}', markersize=6)

ax.set_xlabel('Last-N Tokens', fontsize=12)
ax.set_ylabel('TDNV', fontsize=12)
ax.set_title(f'TDNV Stability vs Last-N Token Selection\nConcept: {concept}, Model: {model_slug}', 
             fontsize=14)
ax.set_yscale('log')
ax.legend(loc='best', ncol=2, fontsize='small')
ax.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = output_dir / f'{concept}_{model_slug}_last_n_trend.pdf'
plt.savefig(plot_path, format='pdf', dpi=150)
print(f'Saved trend plot: {plot_path}')
"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Last-N stability analysis complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  JSON results: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Trend plot:   ${YELLOW}${OUTPUT_DIR}${CONCEPT}_${MODEL_SLUG}_last_n_trend.pdf${NC}"
echo -e "${GREEN}============================================${NC}"
