#!/bin/bash
set -euo pipefail

# Thin shell wrapper around the `plot-heatmap` subcommand of
# `steering_geometry.sweep_evaluation`. All argv parsing and plotting logic
# lives in src/ so this file contains no embedded Python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Default --result to the typical sweep output location when no args are given.
DEFAULT_RESULT="$PROJECT_ROOT/outputs/sweep_evaluation/sentiment/Qwen_Qwen3-1.7B/sweep_results.json"
if [[ $# -eq 0 ]] && [[ -f "$DEFAULT_RESULT" ]]; then
    set -- --result "$DEFAULT_RESULT"
fi

exec uv run python -m steering_geometry.sweep_evaluation plot-heatmap "$@"
