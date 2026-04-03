#!/usr/bin/env bash
# =============================================================================
# Run GPU benchmark on a single GPU
# =============================================================================
# Usage:
#   ./scripts/run_single_gpu.sh                     # default (small-gpt2, bf16)
#   ./scripts/run_single_gpu.sh small-llama auto    # specific model, auto precision
#   ./scripts/run_single_gpu.sh all all             # all models × all precisions
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

MODEL_SIZE="${1:-small-gpt2}"
PRECISION="${2:-bf16}"
BATCH_SIZE="${3:-8}"
SEQ_LENGTH="${4:-2048}"
NUM_STEPS="${5:-100}"
WARMUP_STEPS="${6:-10}"

echo "============================================"
echo "  Transformer Bench — Single GPU"
echo "============================================"
echo "  Model:      $MODEL_SIZE"
echo "  Precision:  $PRECISION"
echo "  Batch size: $BATCH_SIZE"
echo "  Seq length: $SEQ_LENGTH"
echo "  Steps:      $NUM_STEPS (warmup: $WARMUP_STEPS)"
echo "============================================"

# Pin to GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.benchmark \
    --model-size "$MODEL_SIZE" \
    --precision "$PRECISION" \
    --batch-size "$BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --num-steps "$NUM_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --output-dir results
