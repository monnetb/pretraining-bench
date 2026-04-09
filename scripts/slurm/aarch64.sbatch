#!/bin/bash
# =============================================================================
# SLURM job script: transformer-bench on aarch64 partition (aarch64 / GH200)
# =============================================================================
# GH200 nodes have 1 GPU each. This script runs single-GPU benchmarks.
# For multi-node runs, increase --nodes and set NPROC_PER_NODE=1.
#
# Usage:
#   sbatch scripts/slurm/aarch64.sbatch
#   sbatch --export=MODEL_SIZE=8b-llama,PRECISION=fp8-current scripts/slurm/aarch64.sbatch
#   sbatch --nodes=4 --export=ALL,NPROC_PER_NODE=1 scripts/slurm/aarch64.sbatch
# =============================================================================

#SBATCH --job-name=tb-aarch64
#SBATCH --partition=aarch64
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail

# ── Resolve project root (works both from login and compute nodes via NFS) ───
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

# ── Create output directory ──────────────────────────────────────────────────
OUTPUT_DIR="$PROJECT_DIR/results/slurm/aarch64_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect stdout/stderr to the output dir now that it exists
exec > >(tee -a "$OUTPUT_DIR/slurm.out") 2> >(tee -a "$OUTPUT_DIR/slurm.err" >&2)

echo "============================================"
echo "  Transformer Bench — SLURM (aarch64 / GH200)"
echo "============================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Arch:       $(uname -m)"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project:    $PROJECT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "============================================"

# ── Auto-setup venv if needed (aarch64 needs its own wheels) ─────────────────
SCRIPT_DIR="$PROJECT_DIR/scripts"
_ARCH="$(uname -m)"
_NEED_VENV=true

if [ -d "$PROJECT_DIR/.venv-${_ARCH}" ]; then
    _NEED_VENV=false
elif [ -d "$PROJECT_DIR/.venv" ]; then
    # Check if the generic .venv has packages built for this architecture
    # (can't use platform.machine() — venv python symlinks to /bin/python3
    #  which always reports the host arch, not the package arch)
    _test_so="$(find "$PROJECT_DIR/.venv/lib" -name '*.so' -type f 2>/dev/null | head -n 1)"
    if [ -n "$_test_so" ]; then
        _so_info="$(file -b "$_test_so" 2>/dev/null)"
        case "$_ARCH" in
            aarch64) echo "$_so_info" | grep -qi "aarch64\|arm" && _NEED_VENV=false ;;
            x86_64)  echo "$_so_info" | grep -qi "x86.64" && _NEED_VENV=false ;;
        esac
    fi
    if [ "$_NEED_VENV" = true ]; then
        echo "WARNING: .venv exists but contains packages for a different architecture (current: $_ARCH)."
    fi
fi

if [ "$_NEED_VENV" = true ]; then
    echo "ERROR: No compatible virtual environment found for $_ARCH." >&2
    echo "       The existing .venv contains x86_64 packages." >&2
    echo "       Please create an aarch64 venv from a node with internet access:" >&2
    echo "         srun -p aarch64 --exclusive --gpus-per-node=1 --time=00:30:00 \\" >&2
    echo "           $SCRIPT_DIR/setup_venv.sh --arch-suffix" >&2
    echo "       Or use an interactive session:" >&2
    echo "         salloc -p aarch64 --exclusive --gpus-per-node=1 --time=00:30:00" >&2
    echo "         ./scripts/setup_venv.sh --arch-suffix" >&2
    exit 1
fi

# ── Shared environment setup (includes aarch64 NCCL auto-config) ─────────────
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_setup.sh"

# ── Configurable parameters (override via sbatch --export) ───────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL_SIZE="${MODEL_SIZE:-small-llama}"
PRECISION="${PRECISION:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
NUM_STEPS="${NUM_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"

echo "  Model:       $MODEL_SIZE"
echo "  Precision:   $PRECISION"
echo "  Batch/GPU:   $BATCH_SIZE"
echo "  Seq length:  $SEQ_LENGTH"
echo "  Steps:       $NUM_STEPS (warmup: $WARMUP_STEPS)"
echo "  GPUs/node:   $NPROC_PER_NODE"
echo "============================================"

# ── Extra benchmark args (e.g. --activation-checkpointing, --use-compile) ────
EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

# ── Run benchmark ────────────────────────────────────────────────────────────
if [ "${SLURM_NNODES:-1}" -gt 1 ] || [ "$NPROC_PER_NODE" -gt 1 ]; then
    # Multi-GPU / multi-node via torchrun
    NNODES="${SLURM_NNODES:-1}"
    NODE_RANK="${SLURM_NODEID:-0}"
    MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
    MASTER_PORT="${MASTER_PORT:-29500}"

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
        --output-dir "$OUTPUT_DIR" \
        $EXTRA_ARGS
else
    # Single GPU (default for GH200 with 1 GPU/node)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    python -m src.benchmark \
        --model-size "$MODEL_SIZE" \
        --precision "$PRECISION" \
        --batch-size "$BATCH_SIZE" \
        --seq-length "$SEQ_LENGTH" \
        --num-steps "$NUM_STEPS" \
        --warmup-steps "$WARMUP_STEPS" \
        --output-dir "$OUTPUT_DIR" \
        $EXTRA_ARGS
fi

echo "============================================"
echo "  Job complete. Results in: $OUTPUT_DIR"
echo "============================================"
