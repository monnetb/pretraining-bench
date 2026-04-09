#!/usr/bin/env bash
# =============================================================================
# Smoke test harness for transformer-bench
# =============================================================================
# Validates that the benchmark tool works correctly on the current platform.
# Runs a series of short tests and checks results for sanity.
#
# Usage:
#   ./scripts/validate.sh              # single-GPU tests only
#   ./scripts/validate.sh --multi-gpu  # also run 2-GPU FSDP test (needs 2+ GPUs)
#
# Exit 0 = all tests pass, non-zero = at least one failure.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Shared environment setup ─────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_setup.sh"

# ── Parse args ───────────────────────────────────────────────────────────────
MULTI_GPU=false
for arg in "$@"; do
    case "$arg" in
        --multi-gpu) MULTI_GPU=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Test infrastructure ──────────────────────────────────────────────────────
PASSED=0
FAILED=0
SKIPPED=0
RESULTS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tb-validate-XXXXXX")

pass() { echo "  ✓ PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  ✗ FAIL: $1"; echo "    $2"; FAILED=$((FAILED + 1)); }
skip() { echo "  ○ SKIP: $1 — $2"; SKIPPED=$((SKIPPED + 1)); }

run_bench() {
    # Run benchmark and capture output. Returns 0 on success.
    local test_name="$1"; shift
    local logfile="$RESULTS_DIR/${test_name}.log"
    if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
       python -m src.benchmark "$@" --output-dir "$RESULTS_DIR" \
       > "$logfile" 2>&1; then
        return 0
    else
        return 1
    fi
}

check_json() {
    # Validate that the most recent JSON file in RESULTS_DIR has sane values.
    local json_file
    json_file=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)
    if [ -z "$json_file" ]; then
        echo "no JSON output found"
        return 1
    fi

    python3 -c "
import json, sys
with open('$json_file') as f:
    data = json.load(f)
results = data.get('results', [])
if not results:
    print('no results in JSON'); sys.exit(1)
for r in results:
    tok = r.get('tokens_per_second', 0)
    mfu = r.get('mfu', 0)
    mem = r.get('peak_memory_gb', 0)
    if tok <= 0:
        print(f'tokens_per_second={tok} <= 0'); sys.exit(1)
    if mfu <= 0 or mfu > 1.0:
        print(f'mfu={mfu} out of range (0, 1]'); sys.exit(1)
    if mem <= 0:
        print(f'peak_memory_gb={mem} <= 0'); sys.exit(1)
print('JSON sanity check OK')
" 2>&1
}

echo "============================================"
echo "  Transformer Bench — Validation"
echo "============================================"
echo "  Platform:  $(uname -m)"
echo "  Python:    $(python3 --version 2>&1)"
echo "  Temp dir:  $RESULTS_DIR"
echo "  Multi-GPU: $MULTI_GPU"
echo "============================================"
echo ""

# ── Test 1: Single-GPU BF16 ─────────────────────────────────────────────────
echo "Test 1: Single-GPU BF16 (small-gpt2)"
if run_bench "test1_bf16" --model-size small-gpt2 --precision bf16 \
     --num-steps 20 --warmup-steps 5 --batch-size 4; then
    pass "BF16 forward/backward"
else
    fail "BF16 forward/backward" "See $RESULTS_DIR/test1_bf16.log"
fi

# ── Test 2: Single-GPU FP8 (auto-skip if not available) ─────────────────────
echo "Test 2: Single-GPU FP8 (small-llama)"
if run_bench "test2_fp8" --model-size small-llama --precision fp8-current \
     --num-steps 20 --warmup-steps 5 --batch-size 4; then
    pass "FP8 current scaling"
else
    # Check if it was a precision-not-available skip vs a real error
    if grep -q "not available" "$RESULTS_DIR/test2_fp8.log" 2>/dev/null; then
        skip "FP8 current scaling" "GPU does not support FP8"
    else
        fail "FP8 current scaling" "See $RESULTS_DIR/test2_fp8.log"
    fi
fi

# ── Test 3: Fused attention ON (default) ─────────────────────────────────────
echo "Test 3: Fused attention enabled"
if NVTE_FUSED_ATTN=1 run_bench "test3_fused_on" --model-size small-gpt2 --precision bf16 \
     --num-steps 10 --warmup-steps 3 --batch-size 4; then
    pass "Fused attention ON"
else
    fail "Fused attention ON" "See $RESULTS_DIR/test3_fused_on.log"
fi

# ── Test 4: Fused attention OFF ──────────────────────────────────────────────
echo "Test 4: Fused attention disabled (--no-fused-attn)"
if run_bench "test4_fused_off" --model-size small-gpt2 --precision bf16 \
     --num-steps 10 --warmup-steps 3 --batch-size 4 --no-fused-attn; then
    pass "Fused attention OFF (--no-fused-attn)"
else
    fail "Fused attention OFF" "See $RESULTS_DIR/test4_fused_off.log"
fi

# ── Test 5: Gradient accumulation ────────────────────────────────────────────
echo "Test 5: Gradient accumulation (4 steps)"
if run_bench "test5_grad_accum" --model-size small-llama --precision bf16 \
     --num-steps 10 --warmup-steps 3 --batch-size 2 \
     --gradient-accumulation-steps 4; then
    pass "Gradient accumulation"
else
    fail "Gradient accumulation" "See $RESULTS_DIR/test5_grad_accum.log"
fi

# ── Test 6: JSON output sanity ───────────────────────────────────────────────
echo "Test 6: JSON output sanity check"
json_result=$(check_json 2>&1)
if [ $? -eq 0 ]; then
    pass "JSON output: $json_result"
else
    fail "JSON output" "$json_result"
fi

# ── Test 7: Multi-GPU FSDP (optional) ───────────────────────────────────────
if [ "$MULTI_GPU" = true ]; then
    echo "Test 7: Multi-GPU FSDP2 (2 GPUs, medium-llama)"
    mgpu_log="$RESULTS_DIR/test7_multi_gpu.log"
    if torchrun --nproc_per_node=2 -m src.benchmark \
         --model-size medium-llama --precision bf16 \
         --num-steps 15 --warmup-steps 5 --batch-size 4 \
         --output-dir "$RESULTS_DIR" > "$mgpu_log" 2>&1; then
        pass "Multi-GPU FSDP2"
    else
        fail "Multi-GPU FSDP2" "See $mgpu_log"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Validation Summary"
echo "============================================"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  Results: $RESULTS_DIR"
echo "============================================"

if [ "$FAILED" -gt 0 ]; then
    echo "VALIDATION FAILED — $FAILED test(s) failed."
    exit 1
fi

echo "ALL TESTS PASSED."
exit 0
