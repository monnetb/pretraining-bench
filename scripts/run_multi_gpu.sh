#!/usr/bin/env bash
# =============================================================================
# Run GPU benchmark with multiple GPUs using FSDP2 via torchrun
# =============================================================================
# Usage:
#   ./scripts/run_multi_gpu.sh                          # 8 GPUs, default config
#   ./scripts/run_multi_gpu.sh 4 large-llama auto       # 4 GPUs, specific config
#   ./scripts/run_multi_gpu.sh 8 70b-llama auto 1 2048 20 5 8  # TP=8 for 70B
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=host0 ./scripts/run_multi_gpu.sh 8 ...
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

# ── Ensure pip-installed CUDA libs (cuBLAS 13, etc.) are visible to the linker ─
_SITE_PKGS="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
for _nvidia_lib in "$_SITE_PKGS"/nvidia/*/lib; do
    [ -d "$_nvidia_lib" ] && export LD_LIBRARY_PATH="${_nvidia_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# ── Work around cuDNN sublibrary loading failures with TE fused attention ─
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"

NPROC_PER_NODE="${1:-8}"
MODEL_SIZE="${2:-medium-llama}"
PRECISION="${3:-auto}"
BATCH_SIZE="${4:-8}"
SEQ_LENGTH="${5:-2048}"
NUM_STEPS="${6:-100}"
WARMUP_STEPS="${7:-10}"
TP_SIZE="${8:-1}"

# Multi-node support (defaults to single node)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================"
echo "  Transformer Bench — Multi-GPU"
echo "============================================"
echo "  GPUs/node:   $NPROC_PER_NODE"
echo "  Nodes:       $NNODES"
echo "  Model:       $MODEL_SIZE"
echo "  Precision:   $PRECISION"
echo "  Batch/GPU:   $BATCH_SIZE"
echo "  Seq length:  $SEQ_LENGTH"
echo "  Steps:       $NUM_STEPS (warmup: $WARMUP_STEPS)"
echo "  TP size:     $TP_SIZE"
echo "============================================"

# Build extra args
EXTRA_ARGS=""
if [ "$TP_SIZE" -gt 1 ]; then
    EXTRA_ARGS="--tp-size $TP_SIZE"
fi

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m src.benchmark \
    --model-size "$MODEL_SIZE" \
    --precision "$PRECISION" \
    --batch-size "$BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --num-steps "$NUM_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    $EXTRA_ARGS \
    --output-dir results
