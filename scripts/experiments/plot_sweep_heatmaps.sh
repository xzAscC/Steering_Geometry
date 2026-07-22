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

# Pass inputs as argv (not interpolated source) so paths with quotes / shell
# metacharacters cannot break the embedded Python.
uv run python -u -c "
import sys

from steering_geometry.sweep_evaluation import load_sweep_result_json, plot_paper_sweep_heatmap

result_json, output_dir, formats_csv = sys.argv[1:4]
result = load_sweep_result_json(result_json)
formats = [fmt for fmt in formats_csv.split(',') if fmt]
for path in plot_paper_sweep_heatmap(result, output_dir=output_dir, formats=formats):
    print(f'Saved: {path}')
" "$RESULT_JSON" "$OUTPUT_DIR" "$FORMATS"
