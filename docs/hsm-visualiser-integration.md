# HSM Visualiser API — integration guide (Sheffield Bats)

This document summarises how this repository’s outputs connect to the **HSM Visualiser** HTTP API: authentication, **environmental** (driver) COGs, **suitability** COGs per model, metadata (`feature_band_names`), and common pitfalls. It is written for modellers and tooling authors.

**Assumptions**

- API base URL is configurable (e.g. `http://127.0.0.1:8000` in development).
- OpenAPI lives at `{BASE_URL}/openapi.json`; interactive docs at `{BASE_URL}/docs`.
- Admin routes require a **Bearer token** (see [Authentication](#authentication)).

---

## 1. Concepts

| Concept | Role |
|--------|------|
| **Project** | Holds the shared **multi-band environmental COG**, **band definitions** (machine `name`, display `label`, `description`), and optional **explainability background** sample. |
| **Model** | One **species × activity** entry: **suitability COG** (single band, probability-style surface), optional **pickled estimator**, and **`metadata.analysis.feature_band_names`** (ordered list aligned with the model’s feature matrix). |
| **CRS** | Rasters accepted by the API are expected in **EPSG:3857** (Web Mercator). This repo’s modelling stack uses **EPSG:27700** (`evs-to-model.tif`, predictions) — you must **reproject** before upload (see [Reprojecting to EPSG:3857](#4-reprojecting-to-epsg3857)). |
| **COG** | Uploads should be valid **Cloud Optimized GeoTIFFs**. Use `rio-cogeo` (this repo exposes `sdm.raster.io.translate_to_cog`). |

---

## 2. Authentication

Exchange email/password for Firebase **ID token** and use it on admin routes.

```http
POST {BASE_URL}/auth/token
Content-Type: application/json
```

```json
{
  "email": "YOUR_EMAIL",
  "password": "YOUR_PASSWORD",
  "admin_only": true
}
```

Response (fields you need):

- `id_token` — send as `Authorization: Bearer <id_token>`
- `refresh_token`, `expires_in` — for renewing sessions in long scripts

**Example (shell)**

```bash
export BASE_URL="http://127.0.0.1:8000"
export TOKEN="$(
  curl -sS -X POST "${BASE_URL}/auth/token" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"${HSM_EMAIL}\",\"password\":\"${HSM_PASSWORD}\",\"admin_only\":true}" \
  | jq -r '.id_token'
)"
```

Do **not** commit real credentials. Prefer environment variables or a secret manager.

---

## 3. Resolve the catalog project

List projects and pick **Yorkshire HSM** (or your target) by `name`, then read `id`.

```bash
export PROJECT_ID="$(
  curl -sS "${BASE_URL}/projects" \
  | jq -r '.[] | select(.name=="Yorkshire HSM") | .id'
)"
```

Use `PROJECT_ID` as `project_id` when creating/updating **models**.

---

## 4. Reprojecting to EPSG:3857

**Why:** The API validates CRS; **EPSG:27700** uploads are rejected with a structured error (e.g. `COG_CRS_MISMATCH`).

**Environmental stack (full merge used for training)**

- **Source in this repo:** `data/evs/evs-to-model.tif` (see `config.yml` → `paths.ev_tiff`).
- **Recommended pipeline:** `gdalwarp` to 3857 (GeoTIFF), then **COG** encode (e.g. `translate_to_cog`).

```bash
# Step A — warp to Web Mercator (GeoTIFF, compressed)
gdalwarp -overwrite \
  -t_srs EPSG:3857 \
  -r bilinear \
  -multi \
  -co COMPRESS=DEFLATE \
  -co TILED=YES \
  -co BIGTIFF=IF_NEEDED \
  data/evs/evs-to-model.tif \
  data/evs/evs-to-model-epsg3857.tif
```

```bash
# Step B — COG (from repo root, Python env with rio-cogeo)
uv run python -c "
from pathlib import Path
from sdm.raster.io import translate_to_cog
translate_to_cog(
    Path('data/evs/evs-to-model-epsg3857.tif'),
    Path('data/evs/evs-to-model-epsg3857-cog.tif'),
    profile='deflate',
)
"
```

**Note:** Using `gdalwarp … -of COG` in one step has produced **empty** rasters in some GDAL builds; the **two-step** warp → COG is safer here.

**Suitability (per-model prediction)**

- **Source:** `data/sdm_predictions/prediction_<model_id>.tif` where `<model_id>` comes from `get_model_id([latin_name, activity_type])` (lowercase, spaces → underscores), e.g. `nyctalus_noctula_roost`.
- Apply the same **warp → COG** pattern to that file before `POST`/`PUT` **models**.

---

## 5. Environmental COG on the project

### 5.1 Upload / replace driver COG

```http
PUT {BASE_URL}/projects/{PROJECT_ID}
Authorization: Bearer {TOKEN}
Content-Type: multipart/form-data
```

| Part | Purpose |
|------|--------|
| `file` | Multi-band environmental **COG** (EPSG:3857). |
| `infer_band_definitions` | String `true` to **infer** machine `name`s from GDAL band descriptions (omit explicit JSON). Use `false` if you send `environmental_band_definitions`. |

```bash
curl -sS -X PUT "${BASE_URL}/projects/${PROJECT_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@data/evs/evs-to-model-epsg3857-cog.tif" \
  -F "infer_band_definitions=true"
```

After a successful upload, the service may regenerate **explainability** artefacts (e.g. background Parquet). If the COG has **no valid pixels** in sample areas, the `PUT` can fail with an error about explainability sampling — use a real extent or adjust sampling settings on the API side.

### 5.2 Labels and descriptions (PATCH)

Human-facing **`label`** and **`description`** per machine `name`:

```http
PATCH {BASE_URL}/projects/{PROJECT_ID}/environmental-band-definitions/labels
Authorization: Bearer {TOKEN}
Content-Type: application/json
```

Body: object keyed by machine `name`, values `{ "label": "...", "description": "..." }` (see API docs; `name` inside the patch object can alias `label`).

This repo keeps a **ready-made patch** aligned to the current EV stack band names:

```bash
curl -sS -X PATCH "${BASE_URL}/projects/${PROJECT_ID}/environmental-band-definitions/labels" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @data/evs/environmental_band_labels_patch.json
```

Source copy for editing: `temp/all-evs-band-display-names.json` (base layers); focal variants (`*_500m`, etc.) are derived when building the patch file.

---

## 6. Suitability model upload / update

### 6.1 Which files from this repo

| Artefact | Path / pattern |
|----------|----------------|
| Suitability (GeoTIFF, 27700) | `data/sdm_predictions/prediction_<model_id>.tif` |
| Suitability (upload — 3857 **COG**) | Produce with warp + `translate_to_cog` (see §4). |
| Pickled pipeline | `data/sdm_models/<model_id>/model.pkl` (see `model_results.csv` → `model_path`; same row has `model_package_dir` and `package.json` for API metadata). |
| **`feature_band_names` order** | Prefer `package.json` → `feature_names` (same directory as `model.pkl`). Alternatively SHAP `data/sdm_predictions/visualization/shap/<Latin_underscored>/<Activity_underscored>/feature_names.json` — JSON **array as-is**. |

### 6.2 Check if the model already exists

```bash
curl -sS "${BASE_URL}/models" -H "Authorization: Bearer ${TOKEN}" \
  | jq '.[] | select(.species=="Nyctalus noctula" and .activity=="Roost")'
```

- **No row** → `POST /models` — OpenAPI marks **`file`** (suitability COG) as **required**; you cannot create a catalog model without uploading a raster.
- **Row exists** → `PUT /models/{model_id}` to replace the COG, **and/or** send **`metadata`** and/or **`serialized_model_file`** only (all parts optional on `PUT`).

### 6.3 Metadata JSON (multipart string)

`metadata` must be a **JSON object** (minified string in multipart) matching **`ModelMetadata`**: at minimum:

```json
{
  "analysis": {
    "feature_band_names": ["terrain_dtm", "..."]
  },
  "card": {
    "title": "Nyctalus noctula — Roost",
    "summary": "MaxEnt suitability (Sheffield Bats)",
    "version": "2026-04-11"
  }
}
```

Every name in `feature_band_names` must exist exactly once in the **project’s** `environmental_band_definitions[].name` list.

### 6.4 Example `PUT` (update existing model)

```bash
export MODEL_ID="f454a8c7-4d80-463b-9dd2-a8176750703b"   # from GET /models
META="$(jq -c . < temp/metadata_nyctalus_noctula_roost_upload.json)"

curl -sS -X PUT "${BASE_URL}/models/${MODEL_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  --form-string "metadata=${META}" \
  -F "file=@data/sdm_predictions/prediction_nyctalus_noctula_roost_epsg3857_cog.tif" \
  -F "serialized_model_file=@data/sdm_models/nyctalus_noctula_roost/model.pkl"
```

### 6.5 Pickle + metadata only (training `package.json`, no new COG)

When the model row already has a suitability COG on the server, **`PUT /models/{model_id}`** can refresh **only** the pickled pipeline and **`ModelMetadata`** — omit **`file`**.

1. Build API metadata from the training package (maps `feature_names` → `analysis.feature_band_names`, copies CV metrics into `card`, embeds full training JSON into string **`extras.*`** fields for traceability):

   ```bash
   META="$(python scripts/build_model_metadata_from_package.py \
     data/sdm_models/myotis_daubentonii_in_flight/package.json)"
   ```

2. Resolve the catalog UUID (example: *Myotis daubentonii* — *In flight*):

   ```bash
   curl -sS "${BASE_URL}/models" | jq -r '.[] | select(.species=="Myotis daubentonii" and .activity=="In flight") | .id'
   ```

3. **PUT** (admin Bearer token required):

   ```bash
   export MODEL_ID="2d788abb-bcbf-4585-b2c9-a9590bb5f33e"   # example from GET /models
   curl -sS -X PUT "${BASE_URL}/models/${MODEL_ID}" \
     -H "Authorization: Bearer ${TOKEN}" \
     --form-string "metadata=${META}" \
     -F "serialized_model_file=@data/sdm_models/myotis_daubentonii_in_flight/model.pkl"
   ```

**Checks before running:** every `feature_band_names` entry must exist on the parent project’s `environmental_band_definitions` (same names as training). Quick Python check:

```bash
python3 - <<'PY'
import json, sys, urllib.request
base, pid = "http://127.0.0.1:8000", "YOUR_PROJECT_UUID"
pkg = json.load(open("data/sdm_models/myotis_daubentonii_in_flight/package.json"))
proj = json.load(urllib.request.urlopen(f"{base}/projects/{pid}"))
names = {b["name"] for b in (proj.get("environmental_band_definitions") or [])}
missing = [f for f in pkg["feature_names"] if f not in names]
print("missing:", missing)
sys.exit(1 if missing else 0)
PY
```

**Unauthenticated test:** omitting `Authorization` returns **`401`** with `{"detail":"Missing bearer token"}` — the multipart shape above is accepted by the server.

### 6.6 Point inspection (map click)

```http
GET {BASE_URL}/models/{MODEL_ID}/point?lng={WGS84_LON}&lat={WGS84_LAT}
```

Use coordinates **inside** the suitability raster extent (transform raster bounds to EPSG:4326 for a test point).

**Explainability:** If the API returns an error such as *explainability model could not be loaded*, the **pickle** may be incompatible with the server’s Python / `sklearn` / `elapid` versions — fix on the **API runtime** or re-export the model in a compatible environment.

---

## 7. Error reference (typical)

| Symptom | Likely cause |
|--------|----------------|
| `COG_CRS_MISMATCH` | Upload is not **EPSG:3857**. |
| `invalid_feature_band_names` / unknown names | `feature_band_names` not in project manifest; **environmental COG** on the project must match training band names (including focal suffixes). |
| Explainability / sampling errors on **project** `PUT` | COG extent mostly **nodata**, or sample size vs valid pixels. |
| `POINT_SAMPLING` / model load errors on **`/point`** | **Pickle** load failure or missing background configuration on the server. |
| `401` / `403` on writes | Missing/invalid **token**, or `admin_only` / admin claim mismatch. |

---

## 8. Quick checklist

1. Obtain **`TOKEN`** (`POST /auth/token`).
2. Resolve **`PROJECT_ID`** (`GET /projects`).
3. Ensure **environmental** COG is **3857 + COG**; **`PUT /projects`** with `infer_band_definitions=true` (or explicit definitions).
4. Optionally **`PATCH …/labels`** using `data/evs/environmental_band_labels_patch.json` (regenerate if band set changes).
5. Run **`sdm predict`** → **`prediction_<model_id>.tif`** → **warp + COG** for suitability.
6. Build **`metadata`** with **`feature_band_names`** from training **`package.json`** (see `scripts/build_model_metadata_from_package.py`) or from SHAP `feature_names.json`.
7. **`POST /models`** (new row): multipart must include **`file`** (COG) plus `project_id`, `species`, `activity`, and usually `metadata` / `serialized_model_file`. **`PUT /models/{id}`** (existing row): any subset of `file`, `metadata`, `serialized_model_file`.
8. Verify with **`GET /models/{id}`** and **`GET …/point`** inside raster bounds.

---

## 9. Related files in this repository

| File | Purpose |
|------|---------|
| `config.yml` | `paths.ev_tiff` → default EV stack path. |
| `data/evs/evs-to-model-epsg3857-cog.tif` | Example full **environmental** COG in 3857 for upload. |
| `data/evs/environmental_band_labels_patch.json` | **PATCH** payload for display labels/descriptions. |
| `temp/all-evs-band-display-names.json` | Canonical text for base band names (edit / extend for new layers). |
| `sdm/raster/io.py` | `translate_to_cog` helper. |
| `sdm/commands/modelling/predict_sdm_models.py` | Prediction paths and `get_model_id` usage. |
| `sdm/commands/modelling/utils.py` | `get_model_id` implementation. |
| `scripts/build_model_metadata_from_package.py` | Build API `ModelMetadata` JSON from `data/sdm_models/<model_id>/package.json`. |

---

## 10. Recap (what we validated in practice)

- **Token endpoint** works for scripted admin access (`POST /auth/token`, `admin_only: true`).
- **CRS** is strictly **EPSG:3857** for both **project** and **model** COG uploads in the tested deployment.
- **`infer_band_definitions=true`** on **project** `PUT` successfully populated **77** band names from GDAL descriptions; **`PATCH …/labels`** then filled **label** / **description**.
- **`feature_band_names`** on a model must match the **project** manifest (full EV stack with focal bands fixed earlier mismatches).
- **`PUT /models`** with **3857 suitability COG**, **metadata**, and **pickle** returned **200**; **`/point`** explainability may still require **runtime alignment** for pickle loading.

When the API changes, refresh this guide from **`{BASE_URL}/openapi.json`** (e.g. `POST /models` **required** fields vs `PUT /models/{model_id}` optional parts, `ModelMetadata` / `ModelAnalysis` descriptions) and adjust examples accordingly.
