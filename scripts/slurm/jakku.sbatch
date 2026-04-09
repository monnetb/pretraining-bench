#!/bin/bash
# =============================================================================
# SLURM job script: transformer-bench on jakku partition (x86 / H100)
# =============================================================================
# Usage:
#   sbatch scripts/slurm/jakku.sbatch
#   sbatch --export=MODEL_SIZE=8b-llama,PRECISION=fp8-current scripts/slurm/jakku.sbatch
#   sbatch --export=ALL,NUM_STEPS=200,TP_SIZE=8 scripts/slurm/jakku.sbatch
# =============================================================================

#SBATCH --job-name=tb-jakku
#SBATCH --partition=jakku
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00

set -euo pipefail

# ── Resolve project root (works both from login and compute nodes via NFS) ───
# SLURM_SUBMIT_DIR is the directory where sbatch was called
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

# ── Create output directory ──────────────────────────────────────────────────
OUTPUT_DIR="$PROJECT_DIR/results/slurm/jakku_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect stdout/stderr to the output dir now that it exists
exec > >(tee -a "$OUTPUT_DIR/slurm.out") 2> >(tee -a "$OUTPUT_DIR/slurm.err" >&2)

echo "============================================"
echo "  Transformer Bench — SLURM (jakku / H100)"
echo "============================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-8}"
echo "  Project:    $PROJECT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "============================================"

# ── Shared environment setup ─────────────────────────────────────────────────
SCRIPT_DIR="$PROJECT_DIR/scripts"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_setup.sh"

# ── Configurable parameters (override via sbatch --export) ───────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_SIZE="${MODEL_SIZE:-medium-llama}"
PRECISION="${PRECISION:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
NUM_STEPS="${NUM_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
TP_SIZE="${TP_SIZE:-1}"

echo "  Model:       $MODEL_SIZE"
echo "  Precision:   $PRECISION"
echo "  Batch/GPU:   $BATCH_SIZE"
echo "  Seq length:  $SEQ_LENGTH"
echo "  Steps:       $NUM_STEPS (warmup: $WARMUP_STEPS)"
echo "  TP size:     $TP_SIZE"
echo "  GPUs/node:   $NPROC_PER_NODE"
echo "============================================"

# ── Build extra args ─────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ "$TP_SIZE" -gt 1 ]; then
    EXTRA_ARGS="--tp-size $TP_SIZE"
fi
# Pass through any additional benchmark flags (e.g. --activation-checkpointing, --use-compile)
if [ -n "${BENCH_EXTRA_ARGS:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS $BENCH_EXTRA_ARGS"
fi

# ── Run benchmark ────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m src.benchmark \
    --model-size "$MODEL_SIZE" \
    --precision "$PRECISION" \
    --batch-size "$BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --num-steps "$NUM_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    $EXTRA_ARGS \
    --output-dir "$OUTPUT_DIR"

echo "============================================"
echo "  Job complete. Results in: $OUTPUT_DIR"
echo "============================================"
