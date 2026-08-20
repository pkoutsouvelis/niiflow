# niiflow

[![CI](https://github.com/pkoutsouvelis/niiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/pkoutsouvelis/niiflow/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/pkoutsouvelis/niiflow/blob/main/LICENSE)
[![PyPI - niiflow-preproc](https://img.shields.io/pypi/v/niiflow-preproc.svg)](https://pypi.org/project/niiflow-preproc/)

Configurable workflows for deep learning-based neuroimaging analyses.

## Layout (level 3 monorepo)

| Distribution (PyPI name) | Import path        | Role |
|--------------------------|--------------------|------|
| `niiflow-train`          | `niiflow.train`    | MONAI + PyTorch Lightning training / finetuning |
| `niiflow-preproc`        | `niiflow.preproc`  | Offline preprocessing (heavy optional deps stay here) |
| `niiflow` (meta)         | `niiflow`          | Depends on both; `niiflow` CLI entry point |

`niiflow-train`, `niiflow-preproc`, and the **meta** `niiflow` package all contribute to the **`niiflow` namespace** (PEP 420): none of them ship `niiflow/__init__.py`. The meta wheel adds only `niiflow/__main__.py` (CLI) so `import niiflow` still works as a namespace import when any member is installed.

Install only what you need:

```bash
pip install niiflow-train
pip install niiflow-preproc
pip install niiflow   # both + CLI
```

## Development ([uv](https://docs.astral.sh/uv/) workspace)

From the repository root:

```bash
uv sync --group dev
```

The workspace root depends on the `niiflow` meta package (workspace member), so a normal sync installs **niiflow-train**, **niiflow-preproc**, and **torch/monai/lightning** into `.venv` without needing `--all-packages`.

```bash
uv run pytest          # includes line coverage for the niiflow namespace
uv run niiflow
./scripts/format.sh   # ruff (fix) → black → docformatter → black → interrogate
```

CI (`.github/workflows/ci.yml`) runs on every push/PR: `black --check`, `ruff check`, and unit tests (`not integration and not slow and not viz`).

Publishing is manual per distribution (`workflow_dispatch`): see `.github/workflows/publish-niiflow-*.yml` (TestPyPI and PyPI). Requires repo secrets `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN`.

Coverage is on by default (`--cov` / `--cov-report=term-missing` in root `pyproject.toml`). Pass `--no-cov` to skip it.

`scripts/format.sh` prepends `.venv/bin` to `PATH` and uses `uv run -- …` when `uv` is available so tool flags (e.g. docformatter `-e`) are not swallowed by uv.

Build wheels for a member:

```bash
uv build --package niiflow-train
uv build --package niiflow-preproc
uv build --package niiflow
```

The root project `niiflow-workspace` is a **virtual workspace** (not published); it pins the workspace and dev dependencies only. Python **3.11+** is required (aligned with NumPy 2.3.5+).

## Documentation of recent structural changes

See `docs/STRUCTURE.md` for a changelog-style list of packaging and layout edits.
