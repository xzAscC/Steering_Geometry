#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULT_JSON=""
OUTPUT_DIR=""
FORMATS="pdf,png"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--result)
            RESULT_JSON="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --formats)
            FORMATS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --result outputs/sweep_evaluation/<concept>/<model>/sweep_results.json [--output docs/figs] [--formats pdf,png]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$RESULT_JSON" ]]; then
    RESULT_JSON="$PROJECT_ROOT/outputs/sweep_evaluation/sentiment/Qwen_Qwen3-1.7B/sweep_results.json"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(dirname "$RESULT_JSON")"
fi

uv run python -u -c "
from steering_geometry.sweep_evaluation import load_sweep_result_json, plot_paper_sweep_heatmap

result = load_sweep_result_json('${RESULT_JSON}')
formats = [fmt for fmt in '${FORMATS}'.split(',') if fmt]
paths = plot_paper_sweep_heatmap(result, output_dir='${OUTPUT_DIR}', formats=formats)
for path in paths:
    print(f'Saved: {path}')
"
