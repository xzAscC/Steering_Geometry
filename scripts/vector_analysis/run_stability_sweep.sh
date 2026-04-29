#!/bin/bash
# =============================================================================
# run_stability_sweep.sh - Stability Sweep Experiment Runner
# =============================================================================
# Runs stability sweep experiments: for each model, extracts DiM directions
# from independently sampled datasets at varying sample sizes, computes
# pairwise cosine similarity, selects best layer, and generates line-plot
# figures.
#
# Efficiency: loads each model ONCE and processes all concepts in a single
# Python process.  Extracts all layers in one forward pass (10x fewer GPU
# ops vs the naive per-layer loop).
#
# Progress: emits a real-time progress bar with ETA for each model.
#
# Usage:
#   ./scripts/vector_analysis/run_stability_sweep.sh [OPTIONS]
#
# Options:
#   -m, --models     Space-separated model names
#   -c, --concepts   Space-separated concept names
#   -n, --n-values   Space-separated sample sizes (default: 100 300 600 1000 3000)
#   -l, --layers     Space-separated layer fractions (default: 0.1 0.2 ... 1.0)
#   -r, --num-runs   Number of independent runs per setting (default: 5)
#   -o, --output     Output directory (default: outputs/stability_sweep)
#   -h, --help       Show this help message
#
# Example:
#   ./scripts/vector_analysis/run_stability_sweep.sh -m "Qwen/Qwen3-1.7B" -c "refusal polite sentiment"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
# MODELS="Qwen/Qwen3-1.7B Qwen/Qwen3-14B allenai/Olmo-3-1025-7B allenai/Olmo-3-1125-32B"
MODELS="Qwen/Qwen3-1.7B allenai/Olmo-3-1025-7B"
CONCEPTS="refusal polite sentiment"
N_VALUES="100 300 600 1000 3000"
LAYERS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
NUM_RUNS=5
OUTPUT_DIR="$PROJECT_ROOT/outputs/stability_sweep"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Progress bar helpers
# =============================================================================

format_duration() {
    local seconds=$1
    local mins=$((seconds / 60))
    local secs=$((seconds % 60))
    if (( mins >= 60 )); then
        local hrs=$((mins / 60))
        mins=$((mins % 60))
        printf "%dh%02dm%02ds" "$hrs" "$mins" "$secs"
    else
        printf "%dm%02ds" "$mins" "$secs"
    fi
}

draw_progress_bar() {
    local current=$1
    local total=$2
    local width=30
    if (( total == 0 )); then
        total=1
    fi
    local pct=$((current * 100 / total))
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local bar=""
    for ((i = 0; i < filled; i++)); do bar+="█"; done
    for ((i = 0; i < empty; i++)); do bar+="░"; done
    printf "%s" "$bar"
}

# =============================================================================
# Parse command line arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -c|--concepts)
            CONCEPTS="$2"
            shift 2
            ;;
        -n|--n-values)
            N_VALUES="$2"
            shift 2
            ;;
        -l|--layers)
            LAYERS="$2"
            shift 2
            ;;
        -r|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --models     Space-separated model names"
            echo "  -c, --concepts   Space-separated concept names"
            echo "  -n, --n-values   Space-separated sample sizes (default: 100 300 600 1000 3000)"
            echo "  -l, --layers     Space-separated layer fractions (default: 0.1 0.2 ... 1.0)"
            echo "  -r, --num-runs   Number of independent runs (default: 5)"
            echo "  -o, --output     Output directory (default: outputs/stability_sweep)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -m \"Qwen/Qwen3-1.7B\" -c \"refusal polite sentiment\""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Prepare Python literals from shell lists
# =============================================================================

n_values_str="[${N_VALUES// /, }]"
layers_str="[${LAYERS// /, }]"

# Build Python list of quoted concept strings: "['refusal', 'polite', 'sentiment']"
concepts_py="["
first=true
for c in $CONCEPTS; do
    if $first; then first=false; else concepts_py+=", "; fi
    concepts_py+="'$c'"
done
concepts_py+="]"

# Count tasks for progress
NUM_MODELS=$(echo "$MODELS" | wc -w)
NUM_CONCEPTS=$(echo "$CONCEPTS" | wc -w)
NUM_N_VALUES=$(echo "$N_VALUES" | wc -w)
TOTAL_STEPS_PER_MODEL=$(( NUM_CONCEPTS * NUM_N_VALUES * NUM_RUNS ))
GRAND_TOTAL=$(( NUM_MODELS * TOTAL_STEPS_PER_MODEL ))

# =============================================================================
# Print header
# =============================================================================

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Stability Sweep Experiment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Models:        ${GREEN}${MODELS}${NC}"
echo -e "Concepts:      ${GREEN}${CONCEPTS}${NC}"
echo -e "N values:      ${GREEN}${N_VALUES}${NC}"
echo -e "Layers:        ${GREEN}${LAYERS}${NC}"
echo -e "Num runs:      ${GREEN}${NUM_RUNS}${NC}"
echo -e "Output dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "Total steps:   ${CYAN}${GRAND_TOTAL}${NC} (${NUM_MODELS} models × ${NUM_CONCEPTS} concepts × ${NUM_N_VALUES} N-values × ${NUM_RUNS} runs)"
echo -e "${BLUE}============================================${NC}"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Run experiments — one Python invocation per model (batch all concepts)
# =============================================================================

SCRIPT_START=$SECONDS
OVERALL_CURRENT=0

CURRENT_MODEL=0
for MODEL in $MODELS; do
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    MODEL_START=$SECONDS

    echo ""
    echo -e "${YELLOW}┌── [${CURRENT_MODEL}/${NUM_MODELS}] Model: ${MODEL}${NC}"
    echo -e "${YELLOW}│${NC}"

    uv run python -u -c "
import sys, re
from pathlib import Path

from steering_geometry.config import StabilitySweepBatchConfig
from steering_geometry.stability_comparison import run_stability_sweep_batch
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

config = StabilitySweepBatchConfig(
    model_name='${MODEL}',
    concepts=${concepts_py},
    n_values=${n_values_str},
    layers=${layers_str},
    num_runs=${NUM_RUNS},
    output_dir='${OUTPUT_DIR}',
)

results = run_stability_sweep_batch(config)

print()
for r in results:
    print(f'  {r.display_concept}: selected_layer={r.selected_layer}')
    for n in sorted(r.per_n_data):
        d = r.per_n_data[n]
        print(f'    N={n}: cos_sim={d[\"mean\"]:.4f} ± {d[\"std\"]:.4f}')
" 2>&1 | while IFS= read -r line; do
        # Intercept PROGRESS lines for the progress bar
        if [[ "$line" == *INFO*PROGRESS* ]]; then
            # Parse: PROGRESS <concept_idx>/<total_concepts> <step>/<total_steps> <concept> N=<n> run=<r>/<runs>
            progress_part="${line##*PROGRESS }"
            concept_idx="${progress_part%%/*}"
            rest="${progress_part#*/}"
            total_concepts="${rest%% *}"
            rest="${rest#* }"
            step="${rest%%/*}"
            rest="${rest#*/}"
            total_steps="${rest%% *}"
            concept="${rest#* }"
            concept="${concept%% *}"

            # Compute overall position
            model_offset=$(( (CURRENT_MODEL - 1) * TOTAL_STEPS_PER_MODEL ))
            concept_offset=$(( (concept_idx - 1) * NUM_N_VALUES * NUM_RUNS ))
            OVERALL_CURRENT=$(( model_offset + concept_offset + step ))

            elapsed=$(( SECONDS - SCRIPT_START ))
            if (( OVERALL_CURRENT > 0 )); then
                remaining=$(( elapsed * (GRAND_TOTAL - OVERALL_CURRENT) / OVERALL_CURRENT ))
            else
                remaining=0
            fi

            bar=$(draw_progress_bar "$OVERALL_CURRENT" "$GRAND_TOTAL")
            pct=$(( OVERALL_CURRENT * 100 / GRAND_TOTAL ))
            elapsed_str=$(format_duration "$elapsed")
            remain_str=$(format_duration "$remaining")

            printf "\r${YELLOW}│${NC} [%s] %3d%% (%d/%d) | %s elapsed | ~%s remaining | %s" \
                "$bar" "$pct" "$OVERALL_CURRENT" "$GRAND_TOTAL" \
                "$elapsed_str" "$remain_str" \
                "$concept"
        else
            # Forward non-progress lines, indented under the box
            echo -e "${YELLOW}│${NC} $line"
        fi
    done

    MODEL_ELAPSED=$(( SECONDS - MODEL_START ))
    echo -e "${YELLOW}│${NC}"
    echo -e "${YELLOW}└── Done in $(format_duration $MODEL_ELAPSED)${NC}"
done

# =============================================================================
# Generate plots from all results
# =============================================================================

echo ""
echo -e "${YELLOW}=== Generating plots ===${NC}"

uv run python -u -c "
from pathlib import Path

from steering_geometry.stability_comparison import load_sweep_results, plot_stability_sweep
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

output_dir = Path('${OUTPUT_DIR}')
all_results = load_sweep_results(output_dir)

from collections import defaultdict
by_concept: dict[str, dict[str, object]] = defaultdict(dict)
for (model_name, concept), result in all_results.items():
    by_concept[result.display_concept][model_name] = result

paths = plot_stability_sweep(dict(by_concept), output_dir=output_dir)
for p in paths:
    print(f'  Saved: {p}')
"

# =============================================================================
# Print completion message
# =============================================================================

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "Total time:       ${CYAN}$(format_duration $TOTAL_ELAPSED)${NC}"
echo ""
echo -e "To view results:"
echo -e "  ls ${OUTPUT_DIR}/*.json"
echo -e "  ls ${OUTPUT_DIR}/*.pdf"
