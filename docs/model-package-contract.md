# Model package contract

This document defines the **portable artefacts** produced by the HSM (habitat suitability modelling) toolchain in this repository, and how a visualiser consumer maps them to catalog storage.

Sheffield Bats is the reference implementation; the layout is intended to work for other species, study areas, and teams.

### Current consumer: [hsm-app](https://github.com/MatthewJWhittle/hsm-app)

The production visualiser stack is **[MatthewJWhittle/hsm-app](https://github.com/MatthewJWhittle/hsm-app)** (API + web UI). It does **not** mount toolchain directories directly — it ingests **uploaded COGs**, **`ModelMetadata` JSON**, and an optional **pickle** via HTTP (`POST`/`PUT /api/models`, project environmental COG routes).

Authoritative API shape: deployed OpenAPI at [`https://hsm-dashboard-dev.web.app/api/openapi.json`](https://hsm-dashboard-dev.web.app/api/openapi.json) (title: *HSM Visualiser API*). The sections below describe the **on-disk contract** any app could implement; [§6](#6-hsm-app-upload-mapping-reference-consumer) records how hsm-app maps it today.

---

## 1. Catalog concepts (portable)

| Concept | Toolchain artefact | Consumer role (hsm-app) |
|--------|-------------------|-------------------------|
| **Project** | Multi-band environmental GeoTIFF (+ optional label patch JSON) | Shared **driver COG** and `environmental_band_definitions[]` on a catalog project |
| **Model** | `{model_id}/` package + suitability prediction COG | One **species × activity** row: suitability COG, `ModelMetadata`, optional serialized estimator |

**CRS policy (hsm-app):** all **environmental** and **suitability** rasters accepted by the API must be **tiled Cloud Optimized GeoTIFFs in EPSG:3857**. Native toolchain outputs are usually **EPSG:27700** — reproject before upload (`sdm export-rasters`, see [§4](#4-prediction-rasters-suitability-cog)).

**Band alignment:** every name in the model’s ordered feature list must appear **exactly once** in the parent project’s `environmental_band_definitions[].name` (hsm-app resolves these to band indices for point inspection / SHAP).

---

## 2. Model package layout (on disk)

Each trained species × activity model is a **directory**:

```
{models_dir}/
├── model_results.csv              # index (recommended)
└── {model_id}/
    ├── model.pkl                  # required — fitted sklearn/elapid pipeline
    └── package.json               # required — metadata and feature manifest
```

### Model identifier (`model_id`)

Derived from `latin_name` and `activity_type`: lowercased, spaces → underscores, joined with `_`.

Examples: `nyctalus_noctula_roost`, `pipistrellus_pipistrellus_in_flight`.

Implementation: `get_model_id()` in `sdm/commands/modelling/utils.py`.

### Index file (`model_results.csv`)

Written by `sdm train`. Key columns:

| Column | Purpose |
|--------|---------|
| `identifier` | Human-readable id (e.g. `Nyctalus noctula_Roost`) |
| `latin_name`, `activity_type` | Taxon and activity — map to hsm-app `species` / `activity` |
| `mean_cv_score`, `std_cv_score` | Cross-validation metrics |
| `model_path` | Path to `model.pkl` |
| `model_package_dir` | Path to the package directory |

---

## 3. `package.json` schema (version 1)

`schema_version` is **`1`**. Consumers should reject unknown major versions until they explicitly support them.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Contract version (`1`) |
| `identifier` | string | Training identifier |
| `model_id` | string | Filesystem-safe directory name |
| `latin_name` | string | Species binomial → hsm-app **`species`** |
| `activity_type` | string | Activity label → hsm-app **`activity`** |
| `feature_names` | string[] | **Ordered** EV band names → hsm-app **`metadata.analysis.feature_band_names`** |
| `artifacts` | object | At minimum `{ "model_pickle": "model.pkl" }` |

### Optional but usual fields

| Field | Type | Description |
|-------|------|-------------|
| `maxent_config` | object | Serialised MaxEnt / elapid hyperparameters |
| `metrics` | object | e.g. `mean_cv_auc`, `std_cv_auc`, `n_presence`, `n_background`, `training_success` → hsm-app **`metadata.card`** via adapter |

### Example (abbreviated)

```json
{
  "schema_version": 1,
  "identifier": "Nyctalus noctula_Roost",
  "model_id": "nyctalus_noctula_roost",
  "latin_name": "Nyctalus noctula",
  "activity_type": "Roost",
  "feature_names": ["terrain_dtm", "terrain_slope", "ceh_broadleaved_500m"],
  "metrics": { "mean_cv_auc": 0.82, "std_cv_auc": 0.04, "n_presence": 120, "n_background": 4000 },
  "artifacts": { "model_pickle": "model.pkl" }
}
```

### Mapping to hsm-app `ModelMetadata`

hsm-app expects multipart field **`metadata`** as JSON matching OpenAPI schema **`ModelMetadata`**:

| `package.json` | hsm-app `ModelMetadata` |
|----------------|-------------------------|
| `feature_names` (ordered) | `analysis.feature_band_names` |
| `latin_name`, `activity_type` | Also sent as top-level form fields `species`, `activity` |
| `metrics.mean_cv_auc` etc. | `card.primary_metric_type` / `card.primary_metric_value` (via adapter) |
| `latin_name`, `activity_type`, `schema_version` | `card.title`, `card.summary`, `card.version` (via adapter) |
| Full training record | `extras.*` string fields (traceability; optional for other consumers) |

Reference adapter in this repo: `scripts/build_model_metadata_from_package.py` → `sdm.utils.hsm_metadata.model_metadata_from_package`.

---

## 4. Prediction rasters (suitability COG)

Produced by `sdm predict` under `{predictions_dir}/` (default `data/sdm_predictions/`).

### Native toolchain outputs

| File | Description |
|------|-------------|
| `all_predictions.tif` | Multi-band stack, one band per model |
| `prediction_{model_id}.tif` | Single-band surface per model (`--split-files`, default) |

- **CRS:** project CRS from `config.yml` (typically **EPSG:27700**)
- **Encoding:** COG by default (`--cog/--no-cog`)
- **Values:** continuous suitability / probability-style scores

### hsm-app delivery requirements

Upload as multipart field **`file`** on `POST /api/models` (required) or `PUT /api/models/{model_id}` (optional):

| Requirement | Detail |
|-------------|--------|
| CRS | **EPSG:3857** only — otherwise **`COG_CRS_MISMATCH`** |
| Format | Tiled **Cloud Optimized GeoTIFF** |
| Bands | Single-band suitability surface per model row |

Export from native predictions:

```bash
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share \
  --output-crs EPSG:3857 \
  --cog
```

Output example: `prediction_nyctalus_noctula_roost_EPSG3857_cog.tif`.

---

## 5. Environmental stack (project driver COG)

Trained against a multi-band GeoTIFF (default `data/evs/evs-to-model.tif`). hsm-app stores this on the **project** as the shared driver COG.

| Requirement | Detail |
|-------------|--------|
| CRS | **EPSG:3857** COG at rest in hsm-app storage |
| Band manifest | `environmental_band_definitions[]`: `{ index, name, label?, description? }` |
| Inference | `infer_band_definitions=true` on upload reads GDAL band descriptions into `name` |
| Labels | Optional `PATCH /api/projects/{id}/environmental-band-definitions/labels` keyed by machine `name` |

Every `feature_names` entry in each model package must match a project `environmental_band_definitions[].name`.

Export the EV stack the same way as suitability rasters (`sdm export-rasters … --output-crs EPSG:3857 --cog`).

---

## 6. hsm-app upload mapping (reference consumer)

Typical publish sequence (details: [hsm-visualiser-integration.md](hsm-visualiser-integration.md), [notebooks/hsm_api_upload_training_packages.ipynb](../notebooks/hsm_api_upload_training_packages.ipynb)):

1. **Project environmental COG** — `POST /api/projects/{project_id}/environmental-cogs` (multipart: `file`, `infer_band_definitions`)
2. **Band labels** (optional) — `PATCH …/environmental-band-definitions/labels`
3. **Per model** — `POST /api/models` or `PUT /api/models/{model_id}`

| Toolchain artefact | hsm-app multipart / JSON field |
|--------------------|--------------------------------|
| (catalog) project UUID | `project_id` |
| `package.json` → `latin_name` | `species` |
| `package.json` → `activity_type` | `activity` |
| EPSG:3857 suitability COG | `file` |
| `build_model_metadata_from_package.py` output | `metadata` (JSON string) |
| `model.pkl` | `serialized_model_file` → stored as **`serialized_model.pkl`** |

**Create vs update:** `POST /api/models` requires `project_id`, `species`, `activity`, and **`file`**. `PUT /api/models/{model_id}` accepts any subset of `file`, `metadata`, `serialized_model_file`, `species`, `activity`, `project_id`.

**Explainability:** Point inspection (`GET /api/models/{id}/point`) needs aligned `feature_band_names`, a loadable **sklearn-centric** pickle (no custom training-repo imports), and project explainability background. Pickle load failures surface as **`EXPLAINABILITY_PICKLE_IMPORT`** / related 422 responses.

---

## 7. `model.pkl` (serialized estimator)

| Aspect | Contract |
|--------|----------|
| Format | Python pickle of fitted sklearn pipeline (MaxEnt via elapid) |
| hsm-app storage | `serialized_model.pkl` under model `artifact_root` |
| Runtime | Must unpickle with sklearn / numpy / scipy only on the API host |
| Features | Selected subset must match `package.json` → `feature_names` order |

Pickle is environment-sensitive. Re-export or align Python/sklearn/elapid versions if explainability fails after upload.

---

## 8. End-to-end checklist

1. **`sdm train`** → `{model_id}/model.pkl` + `package.json` + `model_results.csv`
2. **`sdm predict`** → `prediction_{model_id}.tif` (COG, project CRS)
3. **`sdm export-rasters … --output-crs EPSG:3857 --cog`** → upload-ready COGs
4. **Publish to hsm-app** (or hand off files): project driver COG + per-model suitability COG + `ModelMetadata` + optional pickle

---

## 9. Related documentation

| Document | Purpose |
|----------|---------|
| [hsm-visualiser-integration.md](hsm-visualiser-integration.md) | Step-by-step hsm-app HTTP upload (auth, curl, pitfalls) |
| [hsm-app OpenAPI](https://hsm-dashboard-dev.web.app/api/openapi.json) | Authoritative `ModelMetadata`, COG validation, error codes |
| [CLI reference](../sdm/README.md) | `sdm train`, `sdm predict`, `sdm export-rasters` |
| `scripts/build_model_metadata_from_package.py` | Training package → hsm-app `ModelMetadata` JSON |

When the contract evolves, bump `schema_version` in new `package.json` files and update this document.
