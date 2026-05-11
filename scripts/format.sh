#!/usr/bin/env bash
# Format and lint Python sources from the repository root.
# Requires dev dependencies: `uv sync --group dev` (uses `uv run` when available).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
run ruff check --fix .

echo "Running Docformatter..."
# Scope to `packages/` only: docformatter only handles Python and can error on odd files at repo root.
# `--black` matches Black’s wrapping; place paths before `-e` so argparse does not treat `.` as an exclude.
run docformatter -r --in-place --black packages

echo "Running Black..."
run black .

echo "Checking doc coverage..."
# Source trees only (skip per-package tests); config in root `pyproject.toml` under `[tool.interrogate]`.
run interrogate packages/niiflow-core/src packages/niiflow-preproc/src packages/niiflow/src

echo "Done."
