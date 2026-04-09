#!/usr/bin/env bash
# =============================================================================
# Create the Python virtual environment and install project dependencies
# =============================================================================
# Usage:
#   ./scripts/setup_venv.sh              # create .venv and install deps
#   ./scripts/setup_venv.sh --dev        # also install dev extras (ruff, pytest)
#   ./scripts/setup_venv.sh --arch-suffix # create .venv-$(uname -m) instead of .venv
#   ./scripts/setup_venv.sh --dev --arch-suffix  # combine both
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DEV_INSTALL=false
ARCH_SUFFIX=false

for arg in "$@"; do
    case "$arg" in
        --dev) DEV_INSTALL=true ;;
        --arch-suffix) ARCH_SUFFIX=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [ "$ARCH_SUFFIX" = true ]; then
    VENV_DIR="$PROJECT_DIR/.venv-$(uname -m)"
else
    VENV_DIR="$PROJECT_DIR/.venv"
fi

echo "============================================"
echo "  Transformer Bench — Environment Setup"
echo "============================================"
echo "  Architecture: $(uname -m)"
echo "  Venv path:    $VENV_DIR"
echo "============================================"

# ── Create venv if it doesn't exist ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# ── Activate ─────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Upgrade pip ──────────────────────────────────────────────────────────────
echo "Upgrading pip ..."
pip install --upgrade pip --quiet

# ── Install project ──────────────────────────────────────────────────────────
if [ "$DEV_INSTALL" = true ]; then
    echo "Installing project with dev extras ..."
    pip install -e ".[dev]" --quiet
else
    echo "Installing project ..."
    pip install -e . --quiet
fi

echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Or just run the scripts directly — they"
echo "  auto-activate the venv."
echo "============================================"
