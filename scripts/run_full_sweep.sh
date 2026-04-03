#!/usr/bin/env bash
# =============================================================================
# Full benchmark sweep: all models × all available precisions
# =============================================================================
# This runs the complete benchmark matrix on a single GPU.
# For multi-GPU sweeps, modify to use torchrun.
#
# Usage:
#   ./scripts/run_full_sweep.sh                 # all models, 100 steps
#   ./scripts/run_full_sweep.sh 50              # all models, 50 steps
#   ./scripts/run_full_sweep.sh 100 small       # only small models
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Activate virtual environment ─────────────────────────────────────────────
if [ -d "$PROJECT_DIR/.venv" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found. Run ./scripts/setup_venv.sh first." >&2
    exit 1
fi

NUM_STEPS="${1:-100}"
SIZE_FILTER="${2:-all}"   # "all", "small", "medium", "large", "xlarge"
BATCH_SIZE="${3:-8}"
SEQ_LENGTH="${4:-2048}"
WARMUP_STEPS="${5:-10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Determine which models to run
case "$SIZE_FILTER" in
    all)
        MODELS="small-gpt2,small-llama,medium-gpt2,medium-llama,large-gpt2,large-llama,xlarge-gpt2,xlarge-llama"
        ;;
    small)
        MODELS="small-gpt2,small-llama"
        ;;
    medium)
        MODELS="medium-gpt2,medium-llama"
        ;;
    large)
        MODELS="large-gpt2,large-llama"
        ;;
    xlarge)
        MODELS="xlarge-gpt2,xlarge-llama"
        ;;
    *)
        MODELS="$SIZE_FILTER"
        ;;
esac

echo "============================================"
echo "  Transformer Bench — Full Sweep"
echo "============================================"
echo "  Models:     $MODELS"
echo "  Precisions: all (auto-detected)"
echo "  Steps:      $NUM_STEPS (warmup: $WARMUP_STEPS)"
echo "  Batch size: $BATCH_SIZE"
echo "  Seq length: $SEQ_LENGTH"
echo "============================================"

python -m src.benchmark \
    --sweep \
    --sweep-models "$MODELS" \
    --batch-size "$BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --num-steps "$NUM_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --output-dir results
