#!/usr/bin/env bash
# =============================================================================
# Create the Python virtual environment and install project dependencies
# =============================================================================
# Usage:
#   ./scripts/setup_venv.sh              # create .venv and install deps
#   ./scripts/setup_venv.sh --dev        # also install dev extras (ruff, pytest)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
DEV_INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --dev) DEV_INSTALL=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "  Transformer Bench — Environment Setup"
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
