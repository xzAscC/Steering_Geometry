#!/bin/bash
# =============================================================================
# plot_stability_sweep.sh - Plot Stability Sweep Results
# =============================================================================
# Loads saved JSON results from a stability sweep and generates line-plot
# figures for a specific layer fraction.
#
# Usage:
#   ./scripts/vector_analysis/plot_stability_sweep.sh [OPTIONS]
#
# Options:
#   -o, --output-dir   Directory with saved JSON results (default: outputs/stability_sweep)
#   -l, --layer        Layer fraction to plot (default: 0.5)
#   -h, --help         Show this help message
#
# Example:
#   ./scripts/vector_analysis/plot_stability_sweep.sh -l 0.7
#   ./scripts/vector_analysis/plot_stability_sweep.sh -o outputs/stability_sweep -l 0.4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
OUTPUT_DIR="$PROJECT_ROOT/outputs/stability_sweep"
LAYER="0.5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Usage
# =============================================================================

usage() {
    head -n 19 "$0" | tail -n +2 | sed 's/^# \?//'
    exit 0
}

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--layer)
            LAYER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Print header
# =============================================================================

echo ""
echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}Stability Sweep Plotting${NC}"
echo -e "${YELLOW}============================================${NC}"
echo -e "  Output dir:  ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  Plot layer:  ${CYAN}${LAYER}${NC}"
echo ""

# =============================================================================
# Generate plots from saved results
# =============================================================================

uv run python -u -c "
from pathlib import Path
from collections import defaultdict

from steering_geometry.stability_comparison import load_sweep_results, plot_stability_sweep
from steering_geometry.utils import configure_logging

configure_logging(level='INFO')

output_dir = Path('${OUTPUT_DIR}')
layer = float('${LAYER}')

all_results = load_sweep_results(output_dir)

if not all_results:
    print('No results found in', output_dir)
    exit(1)

by_concept: dict[str, dict[str, object]] = defaultdict(dict)
for (model_name, concept), result in all_results.items():
    by_concept[result.display_concept][model_name] = result

paths = plot_stability_sweep(dict(by_concept), output_dir=output_dir, plot_layer=layer)
for p in paths:
    print(f'  Saved: {p}')
"

# =============================================================================
# Print completion message
# =============================================================================

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Plotting Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Plots saved to: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
