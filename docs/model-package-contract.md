# Model package contract

This document defines the **portable artefacts** produced by the HSM (habitat suitability modelling) toolchain in this repository. Any visualiser or downstream service can consume these files without importing project code or sharing configuration — the contract is files on disk plus a versioned JSON schema.

Sheffield Bats is the reference implementation; the layout is intended to work for other species, study areas, and teams.

---

## 1. Package layout

Each trained species × activity model is a **directory** (a *model package*):

```
{models_dir}/
├── model_results.csv              # index of all packages (optional but recommended)
└── {model_id}/
    ├── model.pkl                  # required — fitted sklearn/elapid pipeline
    └── package.json               # required — metadata and feature manifest
```

### Model identifier (`model_id`)

Derived from `latin_name` and `activity_type`:

- lowercased
- spaces → underscores
- joined with `_`

Examples: `nyctalus_noctula_roost`, `pipistrellus_pipistrellus_in_flight`.

Implementation: `get_model_id()` in `sdm/commands/modelling/utils.py`.

### Index file (`model_results.csv`)

Written by `sdm train`. One row per model with at least:

| Column | Purpose |
|--------|---------|
| `identifier` | Human-readable id (e.g. `Nyctalus noctula_Roost`) |
| `latin_name`, `activity_type` | Taxon and activity label |
| `mean_cv_score`, `std_cv_score` | Cross-validation metrics |
| `n_presence`, `n_background` | Training counts |
| `model_path` | Path to `model.pkl` inside the package directory |
| `model_package_dir` | Path to the package directory |

Consumers can discover packages via this CSV or by globbing `{models_dir}/*/package.json`.

---

## 2. `package.json` schema (version 1)

`schema_version` is currently **`1`**. Visualisers should reject unknown major versions until they explicitly support them.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Contract version (`1`) |
| `identifier` | string | Training identifier (species + activity) |
| `model_id` | string | Filesystem-safe directory name |
| `latin_name` | string | Species binomial, e.g. `Nyctalus noctula` |
| `activity_type` | string | Activity label, e.g. `Roost`, `In flight` |
| `feature_names` | string[] | **Ordered** list of environmental variable band names used at training time. Must align with the model’s feature matrix and with band names in the shared environmental stack. |
| `artifacts` | object | At minimum `{ "model_pickle": "model.pkl" }` |

### Optional but usual fields

| Field | Type | Description |
|-------|------|-------------|
| `maxent_config` | object | Serialised MaxEnt / elapid hyperparameters |
| `metrics` | object | Training metrics, e.g. `mean_cv_auc`, `std_cv_auc`, `n_presence`, `n_background`, `training_success` |

### Example (abbreviated)

```json
{
  "schema_version": 1,
  "identifier": "Nyctalus noctula_Roost",
  "model_id": "nyctalus_noctula_roost",
  "latin_name": "Nyctalus noctula",
  "activity_type": "Roost",
  "feature_names": ["terrain_dtm", "terrain_slope", "ceh_broadleaved_500m"],
  "maxent_config": { "feature_types": ["linear", "quadratic"], "n_cpus": 4 },
  "metrics": {
    "mean_cv_auc": 0.82,
    "std_cv_auc": 0.04,
    "n_presence": 120,
    "n_background": 4000,
    "training_success": true
  },
  "artifacts": { "model_pickle": "model.pkl" }
}
```

### Mapping to visualiser metadata

`feature_names` maps directly to consumer fields such as `metadata.analysis.feature_band_names`. This repository provides `scripts/build_model_metadata_from_package.py` as a **reference adapter** for one HTTP API shape; other visualisers may map fields differently as long as band order and names are preserved.

---

## 3. `model.pkl`

- **Format:** Python `pickle` of the fitted sklearn pipeline (MaxEnt via elapid).
- **Coupling:** Pickle is environment-sensitive (Python, sklearn, elapid versions). Consumers that load pickles must run a compatible stack or retrain/export in a shared runtime.
- **Features at inference:** The pipeline’s selected feature subset must match `package.json` → `feature_names` order.

---

## 4. Prediction rasters

Produced by `sdm predict` under `{predictions_dir}/` (default `data/sdm_predictions/`).

### Native outputs (toolchain default)

| File | Description |
|------|-------------|
| `all_predictions.tif` | Multi-band stack, one band per model |
| `prediction_{model_id}.tif` | Single-band suitability surface per model (when `--split-files`, default) |

**CRS:** project CRS from `config.yml` (typically **EPSG:27700** / British National Grid).

**Encoding:** Cloud Optimized GeoTIFF (COG) by default (`--cog/--no-cog`). Values are continuous suitability / probability-style scores on the environmental grid.

**Band naming:** Per-band filenames use the model id embedded in GDAL band descriptions, e.g. `prediction_nyctalus_noctula_roost.tif`.

**Masking:** Optional clip to study boundary when `--boundary-path` is set.

### Visualiser / web map delivery (EPSG:3857)

Many map services expect **EPSG:3857** (Web Mercator) COGs. Native predictions are **not** guaranteed to be in 3857 — reproject before publishing:

```bash
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share \
  --output-crs EPSG:3857 \
  --cog
```

Export naming adds suffixes from CRS and COG flags, e.g. `prediction_nyctalus_noctula_roost_EPSG3857_cog.tif`.

**Requirements for consumers expecting Web Mercator:**

- CRS: **EPSG:3857**
- Format: valid **COG** (Cloud Optimized GeoTIFF)
- Single band per suitability surface
- Band / file naming should remain traceable to `model_id`

---

## 5. Environmental stack (companion artefact)

Models are trained against a multi-band environmental GeoTIFF (default `data/evs/evs-to-model.tif`). Visualisers that overlay suitability on drivers need:

- The **same band names** as `feature_names` in each `package.json`
- A **shared CRS** policy for delivery (often EPSG:3857 COG for the full EV stack, produced the same way via `sdm export-rasters`)

This contract treats the EV stack as a **separate published artefact** tied to the project/study, not embedded inside each model package.

---

## 6. End-to-end publish checklist

1. **Train** — `sdm train` → `{model_id}/model.pkl` + `package.json` + `model_results.csv`
2. **Predict** — `sdm predict` → `prediction_{model_id}.tif` (COG, project CRS)
3. **Export for maps** — `sdm export-rasters … --output-crs EPSG:3857 --cog` when the consumer requires Web Mercator
4. **Hand off** — model package directory + exported prediction COG(s); optional adapter JSON from `build_model_metadata_from_package.py`

---

## 7. Related documentation

| Document | Purpose |
|----------|---------|
| [HSM Visualiser integration guide](hsm-visualiser-integration.md) | Worked example uploading to one HTTP API (auth, multipart, pitfalls) |
| [CLI reference](../sdm/README.md) | `sdm train`, `sdm predict`, `sdm export-rasters` |
| `scripts/build_model_metadata_from_package.py` | Reference metadata adapter |

When the contract evolves, bump `schema_version` in new `package.json` files and update this document.
