#!/usr/bin/env bash
set -euo pipefail

# Simple virtual environment setup for this project.
# Usage:
#   ./setup.sh            # create .venv if missing, install requirements
#   ./setup.sh --recreate # delete and recreate .venv, reinstall requirements
#
# After running:
#   source .venv/bin/activate

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

RECREATE=false
for arg in "$@"; do
  case "$arg" in
    --recreate)
      RECREATE=true
      shift
      ;;
    *)
      ;;
  esac
done

if $RECREATE && [ -d "$VENV_DIR" ]; then
  echo "Removing existing virtual environment at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

# Prefer system python on macOS; fallback to python3 on PATH
PY_BIN="/usr/bin/python3"
if [ ! -x "$PY_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  else
    echo "Error: python3 not found. Please install Python 3." >&2
    exit 1
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR using $PY_BIN"
  "$PY_BIN" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

PY_VENV_BIN="$VENV_DIR/bin/python"
PIP_VENV_BIN="$VENV_DIR/bin/pip"

echo "Upgrading pip in the virtual environment"
"$PY_VENV_BIN" -m pip install --upgrade pip

if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from $REQ_FILE"
  "$PIP_VENV_BIN" install -r "$REQ_FILE"
else
  echo "Warning: requirements.txt not found at $REQ_FILE. Skipping dependency install."
fi

echo
echo "Virtual environment is ready: $VENV_DIR"
echo "To activate it in your current shell, run:"
echo "  source .venv/bin/activate"

# If the script is sourced, activate automatically for convenience
IS_SOURCED=false
if [ "${BASH_SOURCE[0]-$0}" != "$0" ]; then
  IS_SOURCED=true
elif [ -n "${ZSH_EVAL_CONTEXT:-}" ] && [[ "$ZSH_EVAL_CONTEXT" == *":file"* ]]; then
  IS_SOURCED=true
fi

if $IS_SOURCED; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "(activated)"
fi


