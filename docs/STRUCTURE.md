# Repository structure changelog

This file records structural / packaging changes so they stay easy to audit.

## 2026-05-11 — Level 3 monorepo + uv workspace

- **Root `pyproject.toml`**: Replaced the single publishable `niiflow` project with a **virtual workspace** named `niiflow-workspace` (`version = 0.0.0`). It declares `[tool.uv.workspace]` members and **does not** define a buildable distribution. It lists `niiflow` (the meta member) under `[project.dependencies]` with `[tool.uv.sources] niiflow = { workspace = true }` so `uv sync` installs all three distributions into the dev environment without `--all-packages`.
- **`[dependency-groups] dev`**: Dev tools (`pytest`, `ruff`, `mypy`, `jupyter`, `build`, `twine`, etc.) are attached to the workspace root for `uv sync --group dev`.
- **`packages/niiflow-core`**: New distribution `niiflow-core` (`0.1.0`) with `src/niiflow/core/` (namespace subpackage only — no `niiflow/__init__.py`). Dependencies match the former root project: `lightning`, `monai`, `torch`, `wandb`, `nifti-finder`, `numpy`, `matplotlib`, `PyYAML`.
- **`packages/niiflow-preproc`**: New distribution `niiflow-preproc` (`0.1.0`) with `src/niiflow/preproc/`. Runtime deps kept light: `numpy`, `nibabel`, `PyYAML`. Optional extra `[ants]` is reserved for future ANTs-related pins (currently empty).
- **`packages/niiflow`**: New **meta** distribution, still named `niiflow` on PyPI, depending on `niiflow-core` and `niiflow-preproc`. Uses `[tool.uv.sources]` with `workspace = true` so local development resolves members from the repo. Ships only `niiflow/__main__.py` (plus namespace packaging metadata) and the **`niiflow` console script** — **no** `niiflow/__init__.py`, so `niiflow.core` / `niiflow.preproc` from the other wheels remain visible (a regular `niiflow` package would shadow subpackages).
- **Tests**: Moved to per-package `tests/` (`packages/*/tests/`). Root pytest config in `pyproject.toml` lists all three `testpaths`.
- **`README.md`**: Updated with layout table and `uv` workflow.
- **`.gitignore`**: Added `.uv/`.
- **`requires-python`**: Set to `>=3.11` on the workspace and all members so `numpy>=2.3.5` resolves (NumPy 2.3.5+ does not support Python 3.10).
- **`[tool.black]`**: `target-version` set to `py311` to match.
- **`uv.lock`**: Added at repo root when you run `uv lock` / `uv sync` (commit it for reproducible dev installs).
- **Per-package `README.md`**: Short stubs under `packages/*/` referenced by each member’s `readme = "README.md"` (paths must stay inside the package directory for setuptools).
- **Legacy paths**: Removed empty root `src/` and `tests/` (only contained `.DS_Store` / were unused after the split).

### Formatting / lint script

- **`scripts/format.sh`**: Runs **Ruff** (`check --fix`), **docformatter** (`packages/` only), **Black** (repo `.`), and **interrogate** (package `src/` trees). Dev deps: `black`, `docformatter`, `interrogate` (plus existing `ruff`) in the workspace **`[dependency-groups] dev`**. **`[tool.interrogate]`** lives in the root `pyproject.toml`.

### Publishing note

When releasing to PyPI, replace workspace-only dependency resolution for the meta package with **version pins** compatible with the released `niiflow-core` / `niiflow-preproc` versions (or use your release automation to rewrite `pyproject.toml` / metadata). The `[tool.uv.sources]` block is for local/workspace use.
