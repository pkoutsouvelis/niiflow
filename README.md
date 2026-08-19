# niiflow

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
uv run pytest
uv run niiflow
./scripts/format.sh   # ruff (fix) → docformatter → black → interrogate
```

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
