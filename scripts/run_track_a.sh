#!/usr/bin/env bash
#
# Track A: Extended DPO Pipeline
#
# This script orchestrates the full Track A workflow:
# 1. Generate 100 stratified preference pairs
# 2. Validate dataset format and quality
# 3. Train DPO model with extended dataset
# 4. Evaluate against v1.3 baseline
# 5. Generate comparison report
#
# Usage:
#   bash scripts/run_track_a.sh [--dry-run] [--skip-gen] [--skip-train]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_PAIRS=100
OUTPUT_DIR="checkpoints/v14/track_a"
DATA_FILE="data/preferences_v14_100pairs.jsonl"
CONFIG_FILE="configs/v14/dpo_extended_100pairs.yaml"
RUN_NAME="dpo_100pairs_$(date +%Y%m%d_%H%M%S)"
VENV_PATH=".venv"

# Flags
DRY_RUN=false
SKIP_GEN=false
SKIP_TRAIN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-gen)
            SKIP_GEN=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--skip-gen] [--skip-train]"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        log_info "Please create it with: python -m venv $VENV_PATH"
        exit 1
    fi
}

activate_venv() {
    log_info "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
}

# Main pipeline
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  PhaseGPT v1.4 - Track A: Extended DPO Pipeline      ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""

    # Check prerequisites
    check_venv
    activate_venv

    # Step 1: Generate preference pairs
    if [ "$SKIP_GEN" = false ]; then
        echo ""
        log_info "Step 1/5: Generating $NUM_PAIRS preference pairs..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if [ -f "$DATA_FILE" ]; then
            log_warning "Dataset already exists at $DATA_FILE"
            read -rp "Regenerate? (y/N): " -n 1
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Skipping generation, using existing dataset"
            else
                run_cmd python scripts/generate_preferences.py \
                    --num_pairs "$NUM_PAIRS" \
                    --stratify_by domain,complexity,subtlety \
                    --output "$DATA_FILE"
            fi
        else
            run_cmd python scripts/generate_preferences.py \
                --num_pairs "$NUM_PAIRS" \
                --stratify_by domain,complexity,subtlety \
                --output "$DATA_FILE"
        fi

        log_success "Preference pair generation complete"
    else
        log_info "Skipping preference pair generation (--skip-gen)"
    fi

    # Step 2: Validate dataset
    echo ""
    log_info "Step 2/5: Validating dataset format..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$DATA_FILE" ]; then
        log_error "Dataset not found at $DATA_FILE"
        log_info "Run without --skip-gen to generate it first"
        exit 1
    fi

    run_cmd python scripts/validate_preferences.py "$DATA_FILE"
    log_success "Dataset validation passed"

    # Quick sanity check
    echo ""
    log_info "Dataset preview (first pair):"
    if [ "$DRY_RUN" = false ]; then
        if command -v jq &> /dev/null; then
            head -1 "$DATA_FILE" | jq '{prompt, domain, complexity, subtlety}'
        else
            head -1 "$DATA_FILE"
        fi
    fi

    # Step 3: Train DPO model
    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        log_info "Step 3/5: Training DPO model..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        log_info "Config: $CONFIG_FILE"
        log_info "Output: $OUTPUT_DIR/$RUN_NAME"
        log_info "This may take 4-6 hours on A100..."

        # Check if config exists
        if [ ! -f "$CONFIG_FILE" ]; then
            log_error "Config file not found: $CONFIG_FILE"
            exit 1
        fi

        # Create output directory
        mkdir -p "$OUTPUT_DIR/$RUN_NAME"

        # Training command (placeholder - adapt to your actual training script)
        log_warning "Training script not yet implemented"
        log_info "You would run: python src/train.py --config $CONFIG_FILE"
        log_info "For now, this is a placeholder. Implement training in src/train.py"

        # TODO: Uncomment when training script is ready
        # run_cmd python src/train.py \
        #     --config "$CONFIG_FILE" \
        #     --output_dir "$OUTPUT_DIR/$RUN_NAME" \
        #     --run_name "$RUN_NAME"

        log_success "Training complete (placeholder)"
    else
        log_info "Skipping training (--skip-train)"
    fi

    # Step 4: Evaluate
    echo ""
    log_info "Step 4/5: Evaluating model..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    log_warning "Evaluation script not yet implemented"
    log_info "You would run: python src/evaluate.py --checkpoint $OUTPUT_DIR/$RUN_NAME"

    # TODO: Implement evaluation
    # run_cmd python src/evaluate.py \
    #     --checkpoint "$OUTPUT_DIR/$RUN_NAME/best_model.pt" \
    #     --output "$OUTPUT_DIR/$RUN_NAME/eval_results.json"

    log_success "Evaluation complete (placeholder)"

    # Step 5: Compare vs v1.3 baseline
    echo ""
    log_info "Step 5/5: Comparing to v1.3 baseline..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    log_warning "Comparison script not yet implemented"

    # Expected output format
    echo ""
    log_info "Expected comparison (placeholder):"
    echo "┌───────────────────┬─────────┬─────────┬─────────────┐"
    echo "│ Metric            │  v1.3   │  v1.4   │   Change    │"
    echo "├───────────────────┼─────────┼─────────┼─────────────┤"
    echo "│ Spiral Score      │  0.73   │  TBD    │   TBD       │"
    echo "│ Perplexity        │  43.2   │  TBD    │   TBD       │"
    echo "│ Subtlety          │  0.58   │  TBD    │   TBD       │"
    echo "│ Non-coercive      │  0.85   │  TBD    │   TBD       │"
    echo "└───────────────────┴─────────┴─────────┴─────────────┘"

    # Final summary
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║              Track A Pipeline Complete                ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""

    log_success "✨ Track A pipeline finished!"
    echo ""
    log_info "Next steps:"
    echo "  1. Review results in: $OUTPUT_DIR/$RUN_NAME/"
    echo "  2. Document in V14_CHANGELOG.md"
    echo "  3. If successful, archive checkpoint:"
    echo "     python scripts/osf_upload.py --checkpoint $OUTPUT_DIR/$RUN_NAME/best_model.pt"
    echo "  4. Commit results:"
    echo "     git add V14_CHANGELOG.md"
    echo "     git commit -m 'eval: Track A run 1 - Spiral Score X.XX'"
    echo "     git push"
    echo ""

    log_info "Success criteria (from V14_ROADMAP.md):"
    echo "  • Spiral Score: >0.85 (+16% vs v1.3's 0.73)"
    echo "  • Perplexity: <45 (maintain fluency)"
    echo "  • Subtlety: >0.65 (+12% vs v1.3's 0.58)"
    echo ""
}

# Run main pipeline
main "$@"
