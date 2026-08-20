#!/usr/bin/env bash
# Format and lint Python sources from the repository root.
# Requires dev dependencies: `uv sync --group dev` (uses `uv run` when available).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Member source trees only (avoids nested package .venv/, tests/, and site-packages).
SOURCE_DIRS=(
  packages/niiflow-train/src
  packages/niiflow-preproc/src
  packages/niiflow/src
)

# Use project venv binaries when `uv` is not on PATH (e.g. uv installed only inside .venv).
if [[ -d "$ROOT/.venv/bin" ]]; then
  PATH="$ROOT/.venv/bin:$PATH"
fi

run() {
  if command -v uv >/dev/null 2>&1; then
    # `--` prevents uv from interpreting tool flags (e.g. docformatter `-e`).
    uv run -- "$@"
  else
    "$@"
  fi
}

echo "Running Ruff..."
run ruff check --fix "${SOURCE_DIRS[@]}"

echo "Running Black..."
run black "${SOURCE_DIRS[@]}"

echo "Running Docformatter..."
# Docstrings only; `--black` matches Black’s wrapping. Do not recurse over `packages/`
# (would hit stale per-package `.venv/` trees and third-party site-packages).
# Exit 3 means files were rewritten — that is success for an in-place run. Under
# `set -e` we must not abort before the final Black pass, which reconciles any
# non-docstring multiline strings docformatter may have touched.
df_status=0
run docformatter -r --in-place --black "${SOURCE_DIRS[@]}" || df_status=$?
if [[ "$df_status" -ne 0 && "$df_status" -ne 3 ]]; then
  exit "$df_status"
fi

echo "Running Black..."
run black "${SOURCE_DIRS[@]}"

echo "Checking doc coverage..."
# Explicit config: `uv run interrogate PATHS` does not always discover root pyproject.toml.
run interrogate -c pyproject.toml "${SOURCE_DIRS[@]}"

echo "Done."
