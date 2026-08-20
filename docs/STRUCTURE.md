# Repository structure changelog

This file records structural / packaging changes so they stay easy to audit.

## 2026-08-20 — Per-package PyPI publish workflows + metadata

- **`.github/workflows/publish-niiflow-{preproc,train}-testpypi.yml`**: `workflow_dispatch` builds the package under `packages/…` and uploads to TestPyPI (`secrets.TEST_PYPI_API_TOKEN`).
- **`.github/workflows/publish-niiflow-{preproc,train}.yml`**: on GitHub Release `published`, build → artifact → PyPI (`secrets.PYPI_API_TOKEN`, `skip-existing: true`).
- Workflows live under `.github/workflows/` (GitHub does not load workflows from `packages/`).
- **`packages/niiflow-preproc` / `packages/niiflow-train` `pyproject.toml`**: added `[project.urls]`, classifiers (MIT; Python 3.11+), and keywords. URLs point at the monorepo `pkoutsouvelis/niiflow`.

## 2026-08-20 — Default pytest coverage

- Root `addopts` now enable `pytest-cov` on every `uv run pytest` (`source_pkgs = ["niiflow"]`, `term-missing` report). Pass `--no-cov` to skip.

## 2026-08-20 — GitHub Actions CI

- **`.github/workflows/ci.yml`**: On push/PR, `uv sync --group dev`, then `black --check` + `ruff check` on member `src/` trees, then `pytest -m "not integration and not slow and not viz"`. Omits docformatter and interrogate.

## 2026-08-20 — Rename `niiflow-core` → `niiflow-train`; independent versions

- **`packages/niiflow-core` → `packages/niiflow-train`**: Distribution renamed to `niiflow-train` (`0.1.0`); import path `niiflow.core` → `niiflow.train`.
- **`packages/niiflow`**: Meta package set to `0.1.0`; depends on `niiflow-train` and `niiflow-preproc` (workspace sources updated).
- **`packages/niiflow-preproc`**: Version bumped to `0.3.0` (independent of train/meta).
- Docs, tests, `scripts/format.sh`, and workspace `testpaths` / members updated accordingly.

## 2026-05-11 — Level 3 monorepo + uv workspace

- **Root `pyproject.toml`**: Replaced the single publishable `niiflow` project with a **virtual workspace** named `niiflow-workspace` (`version = 0.0.0`). It declares `[tool.uv.workspace]` members and **does not** define a buildable distribution. It lists `niiflow` (the meta member) under `[project.dependencies]` with `[tool.uv.sources] niiflow = { workspace = true }` so `uv sync` installs all three distributions into the dev environment without `--all-packages`.
- **`[dependency-groups] dev`**: Dev tools (`pytest`, `ruff`, `mypy`, `jupyter`, `build`, `twine`, etc.) are attached to the workspace root for `uv sync --group dev`.
- **`packages/niiflow-core`**: New distribution `niiflow-core` (`0.1.0`) with `src/niiflow/core/` (namespace subpackage only — no `niiflow/__init__.py`). Dependencies match the former root project: `lightning`, `monai`, `torch`, `wandb`, `nifti-finder`, `numpy`, `matplotlib`, `PyYAML`. *(Later renamed to `niiflow-train` / `niiflow.train`; see 2026-08-20.)*
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

- **`scripts/format.sh`**: Runs **Ruff** (`check --fix`), **Black**, **docformatter** (`--black`), **Black** again, then **interrogate** on member `src/` trees. Docformatter exit code `3` (files rewritten) is treated as success so `set -e` does not skip the final Black pass (needed when docformatter touches multiline non-docstring strings such as SQL). Interrogate is invoked with `-c pyproject.toml`. Dev deps: `black`, `docformatter`, `interrogate`, `ruff` in workspace **`[dependency-groups] dev`**. **`[tool.interrogate]`** lives in the root `pyproject.toml`.

### Publishing note

When releasing to PyPI, replace workspace-only dependency resolution for the meta package with **version pins** compatible with the released `niiflow-train` / `niiflow-preproc` versions (or use your release automation to rewrite `pyproject.toml` / metadata). The `[tool.uv.sources]` block is for local/workspace use.
