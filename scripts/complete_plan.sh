#!/usr/bin/env bash
# complete_plan.sh - Move completed plan from .sisyphus/plans/ to docs/exec-plans/completed/
#
# Usage:
#   ./scripts/complete_plan.sh <plan_name>
#   ./scripts/complete_plan.sh steering-concepts-pipeline
#
# This script is designed to be called after a plan from the Oh My OpenCode
# plan builder (stored in .sisyphus/plans/) has been completed.
#
# What it does:
# 1. Moves the plan file from .sisyphus/plans/ to docs/exec-plans/completed/
# 2. Updates PLAN.md to reference the completed plan
# 3. Updates docs/PLANS.md roadmap status

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
SISYPHUS_PLANS="$PROJECT_ROOT/.sisyphus/plans"
COMPLETED_PLANS="$PROJECT_ROOT/docs/exec-plans/completed"
PLAN_MD="$PROJECT_ROOT/PLAN.md"
PLANS_MD="$PROJECT_ROOT/docs/PLANS.md"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $(basename "$0") <plan_name>

Move a completed plan from .sisyphus/plans/ to docs/exec-plans/completed/

Arguments:
    plan_name    Name of the plan file (without .md extension)
                 Example: steering-concepts-pipeline

Options:
    -h, --help   Show this help message

Examples:
    $(basename "$0") steering-concepts-pipeline
    $(basename "$0") agent-engineering-template

EOF
    exit 0
}

# Parse arguments
PLAN_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        *)
            PLAN_NAME="$1"
            shift
            ;;
    esac
done

# Validate plan name
if [[ -z "$PLAN_NAME" ]]; then
    log_error "Plan name is required"
    usage
fi

# Source and destination paths
SOURCE_FILE="$SISYPHUS_PLANS/${PLAN_NAME}.md"
DEST_FILE="$COMPLETED_PLANS/${PLAN_NAME}.md"

# Check if source file exists
if [[ ! -f "$SOURCE_FILE" ]]; then
    log_error "Plan file not found: $SOURCE_FILE"
    log_info "Available plans in .sisyphus/plans/:"
    ls -1 "$SISYPHUS_PLANS"/*.md 2>/dev/null | xargs -n1 basename | sed 's/\.md$//' || echo "  (none)"
    exit 1
fi

# Check if destination already exists
if [[ -f "$DEST_FILE" ]]; then
    log_warn "Plan already exists in completed/: $DEST_FILE"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted"
        exit 0
    fi
fi

# Ensure completed directory exists
mkdir -p "$COMPLETED_PLANS"

# Move the plan
log_info "Moving plan: $SOURCE_FILE -> $DEST_FILE"
mv "$SOURCE_FILE" "$DEST_FILE"

# Update PLAN.md if it exists
if [[ -f "$PLAN_MD" ]]; then
    log_info "Updating PLAN.md..."
    
    # Create new PLAN.md content
    cat > "$PLAN_MD" << EOF
# Current Task

## Completed: $(echo "$PLAN_NAME" | tr '-' ' ' | sed 's/\b\(.\)/\u\1/g')

**Status**: Completed
**Exec Plan**: [docs/exec-plans/completed/${PLAN_NAME}.md](docs/exec-plans/completed/${PLAN_NAME}.md)

### Summary

Plan has been completed and archived.

### Next Task

(Add your next task here when ready)
EOF
    log_info "Updated PLAN.md"
fi

# Update docs/PLANS.md if it exists
if [[ -f "$PLANS_MD" ]]; then
    log_info "Updating docs/PLANS.md..."
    
    # Check if plan is already in the Done section
    if grep -q "${PLAN_NAME}" "$PLANS_MD"; then
        log_info "Plan already referenced in PLANS.md"
    else
        # Add to Done section (simple append for now)
        # In a more sophisticated version, we'd parse and update the proper section
        log_info "Consider manually updating docs/PLANS.md to move the task to 'Done'"
    fi
fi

log_info "${GREEN}✓${NC} Plan completion successful!"
log_info "  Plan archived at: docs/exec-plans/completed/${PLAN_NAME}.md"
log_info ""
log_info "Next steps:"
log_info "  1. Review the archived plan in docs/exec-plans/completed/"
log_info "  2. Update docs/PLANS.md roadmap if needed"
log_info "  3. Commit the changes: git add docs/exec-plans/completed/${PLAN_NAME}.md"

exit 0
