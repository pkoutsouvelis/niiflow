# niiflow-preproc

Offline, config-driven preprocessing workflows for neuroimaging. `niiflow-preproc`
turns a single YAML/JSON config into a reproducible, parallel preprocessing run over
many images, while keeping the heavy/optional scientific dependencies (ANTs, ANTsPyNet,
DuckDB) isolated from the training stack.

At a glance, it lets you:

- **Build pipelines dynamically** from a list (or id-keyed map) of steps — bias-field
correction, skull-stripping, registration, resampling, intensity normalization, QC,
and more — wiring steps together with `ctx.` references and gating any step with an
`enable` flag (e.g. probe a QC result to the rest of the pipeline).
- **Explore active files dynamically** with the `nifti-finder` backend: glob patterns
plus composable filters (table membership, companion-file existence, AND/OR logic) to
select exactly the images you want (e.g. healthy T1w–FLAIR pairs in a BIDS dataset).
- **Stage additional inputs/outputs around each active file**: search for companion
files near an anchor and construct output paths via root discovery (`parent_match`,
`mirror`, `parent_up`) and dynamic references (`{active.stem}`, `{params...}`).
- **Plan, then execute** with a clean separation of phases: discovery + staging produce
a `RunPlan`; execution runs it with structured logging and multi-process
parallelization.
- **Orchestrate end-to-end from the CLI**: one config wires explorer + staging +
pipeline, runs on explicit files and/or directories, and saves/loads plans as DuckDB
for fast reuse.

---

## Installation

`niiflow-preproc` requires **Python 3.11+** and pulls in ANTs-based tooling (`antspyx`,
`antspynet`), `nibabel`, `PyYAML`, and `duckdb`.

```bash
pip install niiflow-preproc
```

This installs the `niiflow-preproc` console script. For development inside the
[uv](https://docs.astral.sh/uv/) workspace, from the repository root:

```bash
uv sync --group dev
uv run niiflow-preproc --help
```

---

## 1. Dynamic pipeline execution

A pipeline is a dictionary with a `steps` entry. Pass it to
`dynamic_pipeline()`, which builds a `Compose` of `PipelineStage` objects and
runs it immediately under a fresh `RuntimeContext`. `steps` may be either:

- an **ordered list** of step specs (position defines order), or
- an **id-keyed mapping** with an optional top-level `order` (used here so steps can
reference each other by id).

Each step spec is `{name, params, save_outputs, verbose}`:

- `name` — a registered stage class (see [Available stages](#available-stages)).
- `params` — keyword inputs to the stage. String values may use `ctx.` references such
as `"ctx.run_id"` or `"ctx.outputs.<step_id>.<output>"`.
- `save_outputs` — per-output paths to persist results to disk.

The example below uses **simple, explicit paths and `ctx.` references** (no staging). A
single QC step (`CheckVoxelSpacing`) runs first, and its boolean `passed` output is
**probed into the rest of the pipeline via the `enable` flag** — when QC fails, the
gated steps are skipped:

```python
from niiflow.preproc.pipelines import dynamic_pipeline

dynamic_pipeline(
    {
        "verbose": True,
        "steps": {
            # QC gate: always runs, publishes `passed` under id "qc".
            "qc": {
                "name": "CheckVoxelSpacing",
                "params": {
                    "image": "/data/sub-01/anat/sub-01_T1w.nii.gz",
                    "expected": [1.0, 1.0, 1.0],
                    "op": "<=",
                    "id": "ctx.run_id",
                    "log": True,
                },
                "save_outputs": {"report": "/data/derivatives/sub-01_qc.json"},
            },
            # Gated by QC: skipped entirely when qc.passed is False.
            "n4": {
                "name": "ANTsBiasFieldCorrection",
                "params": {
                    "image": "/data/sub-01/anat/sub-01_T1w.nii.gz",
                    "enable": "ctx.outputs.qc.passed",
                },
                "save_outputs": {"out_image": "/data/derivatives/sub-01_desc-n4_T1w.nii.gz"},
            },
            "strip": {
                "name": "ANTsBrainExtraction",
                "params": {
                    "image": "/data/sub-01/anat/sub-01_T1w.nii.gz",
                    "modality": "t1",
                    "enable": "ctx.outputs.qc.passed",
                },
                "save_outputs": {
                    "out_image": "/data/derivatives/sub-01_desc-brain_T1w.nii.gz",
                    "brain_mask": "/data/derivatives/sub-01_desc-brain_mask.nii.gz",
                },
            },
        },
        "order": ["qc", "n4", "strip"],
    },
    run_id="sub-01",
)
```

How the `enable` gate works: `enable` is resolved first (including `ctx.` references)
before any other parameter is loaded. If `enable` is `False`, the stage returns without
calling `forward`, writing outputs, or publishing them to the context — and without
materialising other `params`. This pairs naturally with a boolean QC output like
`"ctx.outputs.qc.passed"`, so downstream gated steps may still declare
`ctx.outputs.<upstream>.<output>` inputs even when that upstream step was skipped.

A common pairing is `GetImage` with this gate: it materialises an image into the context
as `out_image`, so pointing `enable` at a QC output and setting
`save_outputs["out_image"]` to a separate folder persists only the images that passed
quality control, e.g. `params={"image": "ctx.outputs.strip.out_image", "enable":
"ctx.outputs.qc.passed"}` with `save_outputs={"out_image":
"/data/passed/sub-01_T1w.nii.gz"}`.

### Available stages

`ANTsBiasFieldCorrection`, `ANTsBrainExtraction`, `ANTsDenoise`,
`ANTsPreprocessBrainImage`, `ANTsRegistration`, `ANTsApplyTransforms`, `ANTsResample`,
`ANTsResampleToTarget`, `Reorient`, `ClampIntensities`, `MinmaxNorm`, `ZTransformNorm`,
`CenterCrop`, `CenterPad`, `CropToMask`, `CropToRange`, `PadToRange`, `CheckDimensions`,
`CheckVoxelSpacing`, `ApplyMask`, `SmoothMask`, `RelabelMask`, `ToNumpy`,
`GetImage`, `Rename`, `Delete`.

---

## 2. Dynamic exploration of active files

When a workflow `plan` input uses **`mode: search`**, an explorer discovers the
*active files* (the canonical anchor per processing unit, e.g. each subject's T1w).
The explorer lives under that search input's `explorer_params`: glob `patterns` and
optional composable `filters`, backed by
`[nifti-finder](https://pypi.org/project/nifti-finder/)`.

The example below discovers **only healthy T1w images that also have a FLAIR companion**
in a BIDS dataset, combining two filters with `AND` logic:

```yaml
inputs:
  mode: search
  roots: /root/of/BIDS/dataset
  explorer_params:
    patterns: "**/anat/*T1w.nii*"
    filters:
      name: ComposeFilter
      kwargs:
        logic: AND
        filters:
          - name: IncludeFromTable
            kwargs:
              table_path: /root/of/BIDS/dataset/participants.tsv
              id_column: dataset_subject_session
              criteria_column: group
              criteria_value: Control
              id_pattern: "{id}"
          - name: IncludeIfFileExists
            kwargs:
              filename_pattern: "*FLAIR.nii*"
```

- `patterns` accepts a single glob or a list of globs.
- `filters` is either a single `{name, kwargs}` filter or a `ComposeFilter` whose
`kwargs.filters` is a list combined with `logic: AND`/`OR`.
- For **resuming** interrupted cohort runs, nifti-finder also provides
`IncludeFromLogs` / `ExcludeFromLogs` filters that read a workflow `status.log`
(see [nifti-finder](https://github.com/pkoutsouvelis/nifti-finder)).

Other `inputs` shapes: an explicit file path (string/`Path`), `mode: from_file`
(path list on disk), or a sequence mixing any of these. Explicit paths bypass
discovery; only `search` roots are explored.

---

## 3. Flexible staging around active files

Staging resolves *additional* inputs and outputs around each active file **without ever
rewriting the active anchor**. A `FileStager` is configured with `pointers` — dotted
paths into the pipeline `params` — each marked `"input"` or `"output"`:

- **input** pointers may be an explicit path or a *search spec* (`root` + `search`),
resolved to the matching companion file(s).
- **output** pointers build a target path from a discovered `root` plus a `name`,
and may reference resolved inputs via `{params...}` and the active file via
`{active.*}`.
- **`allow_overwrite`** defaults to `true`, so re-staging may replace existing
derivative paths. Set it to `false` to treat existing outputs as already done
(see [Continuing from existing runs](#continuing-from-existing-runs)).

`pointers` may also be omitted entirely. `FileStager` then resolves dynamic
references (`{active.*}` / `{params.*}`) throughout `params` without treating any
parameter as a file to locate or materialise — useful when the pipeline already spells
out every path and only needs the placeholders expanded.

Roots can be **discovered** relative to the active file: `parent_up` (N levels up),
`parent_match` (nearest/farthest ancestor matching a pattern), and `mirror` (rewrite one
hierarchy into another, e.g. `rawdata → derivatives`).

The pipeline and staging specs below work together: step parameters use
`{active.path}` and output `save_outputs` hold staging placeholders that
`FileStager` resolves before execution. A QC step gates skull-stripping via
`enable`:

```yaml
pipeline_params:
  verbose: true
  steps:
    qc:
      name: CheckVoxelSpacing
      params:
        image: "{active.path}"
        expected: [1.0, 1.0, 1.0]
        op: "<="
        id: "ctx.run_id"
        log: true
      save_outputs:
        report:
          root:
            mode: parent_match
            value: "anat"
            mirror:
              source: "rawdata"
              target: "derivatives/niiflow/qc"
          name: "{active.stem}_qc_spacing.json"
    strip:
      name: ANTsBrainExtraction
      params:
        image: "{active.path}"
        modality: t1
        enable: "ctx.outputs.qc.passed"
      save_outputs:
        out_image:
          root:
            mode: parent_match
            value: "anat"
            mirror:
              source: "rawdata"
              target: "derivatives/niiflow"
          name: "{active.stem|strip:_T1w}_desc-brain_T1w.nii.gz"
        brain_mask:
          root:
            mode: parent_match
            value: "anat"
            mirror:
              source: "rawdata"
              target: "derivatives/niiflow"
          name: "{active.stem|strip:_T1w}_desc-brain_mask.nii.gz"
  order: [qc, strip]

staging_params:
  stager_name: FileStager
  params:
    pointers:
      steps.qc.params.image: input
      steps.qc.save_outputs.report: output
      steps.strip.params.image: input
      steps.strip.save_outputs.out_image: output
      steps.strip.save_outputs.brain_mask: output
```

For an active file `.../rawdata/sub-01/anat/sub-01_T1w.nii.gz` this stages the T1w as
QC and strip inputs, writes the QC report under `derivatives/niiflow/qc/...`, and
writes skull-stripped outputs under `derivatives/niiflow/...` when QC passes.

Supported references inside specs: `{active.path|name|stem|parent}` and
`{params.<dotted.path>[.path|name|stem|parent]}`, with a `|strip:<suffix>` modifier
(e.g. `{active.stem|strip:_T1w}`). `resolve_results` may be `first`, `single`, or `all`.

`staging_params` accepts a single `{stager_name, params}` dict or a **list** of such
dicts; when a list is provided, stagers run **in order**, each transforming the entries
produced by the previous one (useful for splitting input and output resolution, or
layering multiple stager types as more are added):

```yaml
staging_params:
  - stager_name: FileStager
    params:
      pointers:
        steps.qc.save_outputs.report: output
  - stager_name: FileStager
    params:
      pointers:
        steps.strip.save_outputs.out_image: output
        steps.strip.save_outputs.brain_mask: output
```

### Available stagers

`FileStager`.

---

## 4. Workflows: planning and controlled execution

`DynamicPreprocessingWorkflow` ties everything together. It separates **planning**
(discovery + staging → a `RunPlan`) from **execution** (running the plan), and provides
structured logging and process-based parallelism.

```python
from niiflow.preproc.workflows import DynamicPreprocessingWorkflow, dynamic_workflow

workflow = DynamicPreprocessingWorkflow(
    pipeline_params={...},        # the `steps` spec from section 1/3 (required)
    staging_params={...},         # the stager from section 3 (optional)
    num_workers=1,                # serial by default; "auto" or N>1 for parallel cohort runs
    logs_root="/data/logs",       # writes main.log, status.log, workers.log
    timeout=1800,                 # soft per-entry limit (seconds)
)

# Phase 1 — plan only (discover active files + stage entries, no processing):
plan = workflow.plan(
    [
        {
            "mode": "search",
            "roots": "/data/bids",
            "explorer_params": {"patterns": "**/anat/*T1w.nii*"},
        },
        "/data/extra/sub-99_T1w.nii.gz",
    ]
)
plan.save("/data/plans/run.duckdb")   # persist for reuse

# Phase 2 — execute the prepared plan:
workflow.run_plan(plan)

# Or one-shot orchestration (plan / plan-only / from-plan / dry-run):
dynamic_workflow(
    settings={"pipeline_params": {...}, "staging_params": {...}, "num_workers": 1},
    inputs="/data/extra/sub-99_T1w.nii.gz",
)
```

- **Planning** (`plan`) discovers active files from `InputData` (`search` /
`from_file` / explicit paths), stages per-entry parameters, and returns a `RunPlan`
of `StagedEntry` objects. To plan and execute, call `plan` then `run_plan`, or use
`dynamic_workflow(...)` for a single orchestration entry point (plan /
plan-only / from-plan / dry-run). That driver returns nothing; persist with
`save_plan_to` and reload via `from_plan` / `RunPlan.load` when you need the plan.
The CLI `dynamic_workflow` command uses that driver.
- **Logging** is controlled by `logs_root` and the `main_logs` / `status_logs` /
`worker_logs` / `dev_mode` flags. The status log records one line per entry
(`<active> | SUCCESS`, `... | FAILURE | <traceback>`, `... | TIMEOUT`, or
`... | STAGING_FAILURE | <message>` when staging recorded errors on an entry).
- **Parallelization** defaults to serial (`num_workers=1`). Set `num_workers` to
`"auto"` (CPU count) or an integer `> 1` to run entries in a `ProcessPoolExecutor`.
For parallel cohort runs, also consider limiting per-process ITK/OMP threads to
avoid oversubscription. `timeout` is a soft per-entry limit reported in the status log.
- **RunPlan persistence**: `.duckdb` (recommended, scalable) or `.json` (debug).
Reload with `RunPlan.load(path)` and execute without re-discovering or re-staging.

### Continuing from existing runs

By default, `FileStager` uses `allow_overwrite: true`, so a repeat `execute` may
overwrite derivative outputs. To **continue** after a partial or failed run, narrow
which actives are discovered or which entries pass staging:

**1. Filter actives with `status.log` (explorer)**  
Add an `IncludeFromLogs` or `ExcludeFromLogs` filter under
`inputs.explorer_params.filters` (for a `mode: search` input), pointing at the
workflow `status.log` from a previous run. Typical patterns: exclude actives that
already logged `SUCCESS`, or include only `FAILURE` / `TIMEOUT` lines for a retry
pass. Filter kwargs are documented in
[nifti-finder](https://github.com/pkoutsouvelis/nifti-finder).

```yaml
inputs:
  mode: search
  roots: /root/of/BIDS/dataset
  explorer_params:
    patterns: "**/anat/*T1w.nii*"
    filters:
      name: ExcludeFromLogs
      kwargs:
        log_path: /data/logs/status.log
        # see nifti-finder for status / path-matching options
```

**2. Skip entries whose staged outputs already exist (FileStager)**  
Set `allow_overwrite: false` on `FileStager`. Resolved output pointers that already
exist raise a staging error; at execution those entries are logged as
`STAGING_FAILURE` and skipped. Use `allow_failed_entries: true` so planning continues
for the remaining actives.

```yaml
staging_params:
  stager_name: FileStager
  params:
    allow_overwrite: false
    allow_failed_entries: true
    pointers:
      steps.strip.save_outputs.out_image: output
      # ...
```

**3. Re-execute a saved plan**  
Set `from_plan` in a `dynamic_workflow` config (or call
`workflow.run_plan(RunPlan.load(...))`) to reload a saved plan without
re-discovery or re-staging. Pair this with `status.log` filters when you need a fresh
plan that omits already-finished actives.

### Available workflows

`DynamicPreprocessingWorkflow`.

---

## 5. CLI: end-to-end orchestration

The CLI is a thin registry: each subcommand name is an orchestration driver, and the
config file is that driver's keyword arguments. Today the registered command is
`dynamic_workflow`.

An end-to-end example — BIDS T1w discovery (controls with FLAIR companions), QC-gated
skull-stripping, and staged derivative paths — lives at
`[configs/bids_controls_t1w_flair.yaml](configs/bids_controls_t1w_flair.yaml)`.
Replace placeholder paths (`/root/of/BIDS/...`, `/data/...`) before running.

```bash
# Plan from inputs and execute (config may include save_plan_to):
niiflow-preproc dynamic_workflow configs/bids_controls_t1w_flair.yaml

# Show full tracebacks on error:
niiflow-preproc --debug dynamic_workflow configs/bids_controls_t1w_flair.yaml

# List registered commands:
niiflow-preproc --help
```

Modes belong in the YAML (not CLI flags), for example:

```yaml
# plan only (discover + stage, save, do not execute)
plan_only: true
save_plan_to: /data/plans/controls.duckdb

# dry run (print plan; do not save or execute)
dry_run: true

# execute a previously saved plan (no re-discovery / re-staging)
from_plan: /data/plans/controls.duckdb
# omit inputs / save_plan_to when using from_plan
```

Configs may be `.yaml`, `.yml`, or `.json`. Use `.duckdb` for production-scale plans
you intend to reload, and `.json` for quick inspection.

---

## Project structure

```
niiflow-preproc/
├── pyproject.toml
├── README.md
├── configs/                          # example config (BIDS T1w–FLAIR + QC + strip)
├── tests/
└── src/niiflow/preproc/
    ├── cli/
    │   ├── main.py                   # argparse entry (registered commands + config path)
    │   └── commands.py               # COMMANDS registry / run()
    ├── config/
    │   └── load.py                   # load_config (YAML/JSON → dict)
    ├── data/
    │   ├── explorer_factory.py       # get_data_explorer (nifti-finder + filters)
    │   ├── read_from_file.py         # read_paths_from_file
    │   └── types.py
    ├── staging/
    │   ├── stager.py                 # Stager base, StagedEntry, make_entries
    │   ├── file_stager.py            # FileStager (input/output pointer resolution)
    │   ├── stager_factory.py         # create_stager / discovery
    │   ├── search.py                 # parent_up / parent_match / mirror_root
    │   ├── dynamic.py                # {active.*} / {params...} reference resolution
    │   ├── validation.py
    │   └── types.py                  # Root/Input/Output spec types
    ├── pipelines/
    │   ├── dynamic_pipeline.py       # dynamic_pipeline (build + execute)
    │   └── pipeline_stages/
    │       ├── pipeline_stage.py     # PipelineStage base + RuntimeContext
    │       ├── compose.py            # Compose (ordered stage runner)
    │       ├── bias_field.py         # ANTsBiasFieldCorrection
    │       ├── skull_stripping.py    # ANTsBrainExtraction
    │       ├── registration.py       # ANTsRegistration / ANTsApplyTransforms
    │       ├── resampling.py         # ANTsResample / ANTsResampleToTarget
    │       ├── denoising.py          # ANTsDenoise
    │       ├── intensity_normalization.py  # Clamp / Minmax / ZTransform
    │       ├── croppad.py            # CenterCrop / CenterPad / Crop/Pad-to-range
    │       ├── qc.py                 # CheckDimensions / CheckVoxelSpacing
    │       ├── masks.py              # ApplyMask / SmoothMask / RelabelMask
    │       ├── utility.py            # Reorient / ToNumpy / GetImage / Rename / Delete
    │       └── pipelines.py          # ANTsPreprocessBrainImage
    ├── workflows/
    │   ├── workflow.py               # ProcessingWorkflow / PlannableWorkflow (execution engine)
    │   ├── dynamic_workflow.py       # DynamicPreprocessingWorkflow / dynamic_workflow
    │   ├── workflow_factory.py       # create_workflow / discovery
    │   ├── mixins.py                 # SupportsFileDiscovery / SupportsStaging
    │   ├── types.py                  # InputData / SearchInput / FromFileInput
    │   ├── plan.py                   # RunPlan (.duckdb / .json persistence)
    │   ├── logging_manager.py        # main / status / parallel logging
    │   └── logging_utils.py
    ├── functional/                   # array/image operations (core preprocessing functions)
    └── utils/                        # file + misc helpers
```

