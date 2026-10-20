# GCS data layout (`gs://habitat-suitability-modelling-data`)

Training and tuning read inputs from **`inputs/`** (via `INPUT_DATA_ROOT`, defaulting to `gs://habitat-suitability-modelling-data/inputs` when set). Each **`sdm train`** run writes outputs under **`runs/<UTC>-<id>/`** (via `MODEL_OUTPUT_ROOT`, default bucket root `gs://habitat-suitability-modelling-data`).

Repository paths in `config.yml` are relative keys; with `INPUT_DATA_ROOT` set, they resolve to `inputs/<config path>`.

## Inputs (read by train and/or tune)

| Config key | Local default | Bucket object (under `inputs/`) | Format | Used by |
| --- | --- | --- | --- | --- |
| `paths.occurence_data` | `data/processed/bats-tidy.geojson` | `inputs/data/processed/bats-tidy.geojson` | GeoJSON (points) | **train**, **tune** |
| `paths.boundary` | `data/processed/boundary.geojson` | `inputs/data/processed/boundary.geojson` | GeoJSON (polygon) | **train**, **tune** |
| `paths.ev_tiff` | `data/evs/evs-to-model.tif` | `inputs/data/evs/evs-to-model.tif` | GeoTIFF (multi-band EV stack) | **train**, **tune** (via `/vsigs/`) |
| `paths.background_points` | `data/processed/background-points.geojson` | `inputs/data/processed/background-points.geojson` | GeoJSON | **train** (`setup_training_data` only) |
| `paths.grid_points` | `data/evs/grid-points.parquet` | `inputs/data/evs/grid-points.parquet` | Parquet (optional; derived from EV if missing) | **train** (`setup_training_data`) |
| `paths.model_config` | `model_config.yml` (repo root) | `inputs/model_config.yml` (optional mirror) | YAML | **train**, **tune** (base config; often kept in repo) |
| `paths.variables_config` | `variables_config.yml` | `inputs/variables_config.yml` (optional) | YAML | **train** (fallback roster), **tune** |
| `paths.config_dir` | `data/sdm_config` | `inputs/data/sdm_config/` | YAML per `{model_id}/` | **train** (per-species overrides) |
| `paths.tuning_dir` | `data/sdm_tuning` | `inputs/data/sdm_tuning/` | YAML per `{model_id}/` | **train** when `config_dir` points here; **tune** reads base configs only |

Per-model override layout (under `config_dir` or `tuning_dir`):

```
inputs/data/sdm_config/{model_id}/model_config.yml
inputs/data/sdm_config/{model_id}/variables_config.yml
```

## Outputs (written by train)

Each run: `runs/<UTC>-<hex>/` under the output root.

| Artifact | Relative path | Format |
| --- | --- | --- |
| Model package | `{model_id}/model.pkl` | Pickled sklearn pipeline |
| Package metadata | `{model_id}/package.json` | JSON (metrics, feature list) |
| Optional CV scores | `{model_id}/validation_scores.parquet` | Parquet |
| Suitability cut-offs | `thresholds.csv` / `thresholds.json` (also `{model_id}/thresholds.json`) | Cut-off per model (METHOD.md rule); no locations |
| Run summary | `model_results.csv` | CSV (`model_path`, `model_package_dir` as final `gs://…` URIs) |
| Training snapshot | `training_data.parquet` | Parquet |

Example: `gs://habitat-suitability-modelling-data/runs/20251016T120000Z-a1b2c3d4/model_results.csv`

## Outputs (written by tune)

| Artifact | Config / CLI | Bucket location (typical) | Notes |
| --- | --- | --- | --- |
| Trial configs | `--output-dir` / `paths.tuning_dir` | Local or sync to `inputs/data/sdm_tuning/` | Not auto-uploaded; copy into `inputs/` for train to consume |
| Optuna DB | `--storage` URL | N/A unless external DB | Optional |

## Config paths not used by train/tune I/O

| Config key | Notes |
| --- | --- |
| `paths.raw_data` | Data-prep pipeline only |
| `paths.processed_data` | Parent directory; individual files listed above |
| `paths.models` | Default **output** root when `MODEL_OUTPUT_ROOT` unset (local `data/sdm_models`) |
| `paths.predictions` | **`sdm predict`** only |
| `paths.evs` | Directory of layer prep; train uses merged `ev_tiff` only |
| `config.yml` itself | Loaded from repo checkout (not from bucket) |
| `mlflow.*` | Local/SQLite tracking URI by default |

## Environment variables

| Variable | Purpose |
| --- | --- |
| `INPUT_DATA_ROOT` | Prefix for config-relative input paths (e.g. `gs://habitat-suitability-modelling-data/inputs`) |
| `MODEL_OUTPUT_ROOT` | Bucket or directory root for run folders (e.g. `gs://habitat-suitability-modelling-data`) |

Authentication is standard **gcsfs** / Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS` (no custom checks in application code).

## Code modules

| Module | Role |
| --- | --- |
| `sdm/utils/fs.py` | Storage-agnostic helpers: path joining, existence checks, GDAL `/vsigs/` mapping, downloading remote files for local-only libraries, idempotent byte/text writes, and `staged_directory_context` for publishing a local staging tree to local or remote storage. |
| `sdm/utils/model_runs.py` | Run layout only: `INPUT_DATA_ROOT` / `MODEL_OUTPUT_ROOT` resolution, `runs/<UTC>-<id>/` URIs, and published artifact paths for `model_results.csv`. |
