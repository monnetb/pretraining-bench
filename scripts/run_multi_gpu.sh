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

# ── Shared environment setup (venv, CUDA libs, NCCL, fused attn) ──
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_setup.sh"

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
