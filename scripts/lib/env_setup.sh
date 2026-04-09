#!/usr/bin/env bash
# =============================================================================
# Shared environment setup for transformer-bench
# =============================================================================
# Source this file from run scripts:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/lib/env_setup.sh"
#
# Expects PROJECT_DIR to be set by the caller.
# =============================================================================

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    echo "ERROR: PROJECT_DIR must be set before sourcing env_setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# ── Activate virtual environment ─────────────────────────────────────────────
# Prefer architecture-specific venv (.venv-x86_64 or .venv-aarch64).
# Only fall back to generic .venv if its installed packages match the current arch.
# NOTE: We can't rely on `python3 -c platform.machine()` because the venv
# symlinks to /bin/python3 which always reports the *host* arch, not the arch
# of the pip-installed wheels.  Instead we inspect a compiled .so file.
_ARCH="$(uname -m)"
_VENV_ACTIVATED=false

_check_venv_arch() {
    # Returns 0 (true) if the venv at $1 has packages matching $_ARCH
    local venv_dir="$1"
    # Find any .so in the site-packages to check its ELF architecture
    local _test_so
    _test_so="$(find "$venv_dir/lib" -name '*.so' -type f 2>/dev/null | head -n 1)"
    if [ -z "$_test_so" ]; then
        # No .so files — probably a fresh venv with no packages; allow it
        return 0
    fi
    local _so_arch
    _so_arch="$(file -b "$_test_so" 2>/dev/null)"
    case "$_ARCH" in
        x86_64)  echo "$_so_arch" | grep -qi "x86.64" && return 0 ;;
        aarch64) echo "$_so_arch" | grep -qi "aarch64\|arm" && return 0 ;;
        *)       return 0 ;;  # unknown arch, optimistic
    esac
    return 1
}

if [ -d "$PROJECT_DIR/.venv-${_ARCH}" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv-${_ARCH}/bin/activate"
    _VENV_ACTIVATED=true
elif [ -d "$PROJECT_DIR/.venv" ]; then
    # Check that the generic .venv has packages built for the current architecture
    if _check_venv_arch "$PROJECT_DIR/.venv"; then
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/.venv/bin/activate"
        _VENV_ACTIVATED=true
    else
        echo "WARNING: .venv exists but contains packages for a different architecture (current: $_ARCH). Skipping." >&2
    fi
fi

if [ "$_VENV_ACTIVATED" = false ]; then
    echo "ERROR: No compatible virtual environment found for $_ARCH." >&2
    echo "       Run: ./scripts/setup_venv.sh --arch-suffix" >&2
    return 1 2>/dev/null || exit 1
fi

# ── Ensure pip-installed CUDA libs (cuBLAS 13, etc.) are visible to the linker ─
_SITE_PKGS="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
for _nvidia_lib in "$_SITE_PKGS"/nvidia/*/lib; do
    [ -d "$_nvidia_lib" ] && export LD_LIBRARY_PATH="${_nvidia_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# ── Help cuDNN find its engine sublibraries (pip-installed cuDNN) ────────────
# cuDNN dynamically loads engine sublibraries at runtime.  When installed via
# pip the standard lookup may fail with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
# Setting CUDNN_PATH tells cuDNN where to find them.
if [ -d "$_SITE_PKGS/nvidia/cudnn" ]; then
    export CUDNN_PATH="${CUDNN_PATH:-$_SITE_PKGS/nvidia/cudnn}"
fi

# ── Enable fused attention by default (override with NVTE_FUSED_ATTN=0) ─
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"

# ── CUDA memory allocator: reduce fragmentation for large models ─────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Maximize CUDA kernel pipelining for TP workloads ─────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# ── Platform-aware NCCL configuration ────────────────────────────────────────
# aarch64 systems (Grace Hopper GH200, Grace Blackwell GB300) use NVLink
# instead of InfiniBand for inter-GPU communication.
if [ "$_ARCH" = "aarch64" ]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
fi
