# QGIS review project

In-repo QGIS project for reviewing citizen-science model maps straight from the bucket (issue #127). Project tooling under `qgis/review/` — not part of the `sdm` modelling library or CLI. No download step: every prediction layer uses GDAL `/vsigs/` under `gs://habitat-suitability-modelling-data/…`.

## Open the project

1. Authenticate for GCS (once per machine):

   ```bash
   gcloud application-default login
   ```

2. Open `qgis/review/hsm-review.qgs` in QGIS (plain-text `.qgs`, not `.qgz`).

3. QGIS reads rasters via `/vsigs/…` using Application Default Credentials. If a layer fails to load, confirm ADC and that the path in the [manifest](../qgis/review/manifest.yml) still exists.

## Layout

- **Manifest** (`qgis/review/manifest.yml`): required `versions.main` plus optional `versions.compare`. Each version has a short `label` and per-model `run_folder` + `prediction_file` (optional `cutoff_file`). Top-level `crs` (default `EPSG:27700`) sets prediction/cutoff layer SRS and the project CRS; XYZ basemaps stay `EPSG:3857`.
- **Main** paths use the prediction-version layout from [#104](https://github.com/MatthewJWhittle/shefflied-bats/pull/104): `predictions/<model_id>/<YYYYMMDD>_<sha7>_cfg<hash>/prediction__…__suitability.tif`.
- **Groups** in the project: one top-level group per version (`Main`, `Compare: <label>`), then the same per-species subgroups and layer names so maps line up.
- **Basemaps**: Google Satellite (default; same XYZ as the `hsm-aerial-latest.qgz` style reference), OpenStreetMap, and Carto. Suitability/cutoff layers use the shared plasma 0–1 ramp from that reference at renderer opacity 0.5.
- **Map Themes** (QGIS *View → Map Themes*): `Main` (default), `Compare` when the compare block is present, and per-model themes (`Myotis brandtii Roost — Main` / `… — Compare`) that show only that model’s layers from one version plus the basemap. Layer names include species and version so Main and Compare stay distinguishable in the legend.

## Point compare at another run

1. Edit `versions.compare` in `qgis/review/manifest.yml` — set `label`, and for each model set `run_folder` / `prediction_file` (and `cutoff_file` when present).
2. To drop compare and keep a main-only project, omit `compare` or set `models: {}`.
3. Regenerate:

   ```bash
   uv run python -m qgis.review.generate
   ```

4. Commit both the manifest and the regenerated `.qgs`. CI regenerates from the manifest and fails if the committed project differs, if credentials/tokens appear, or if a layer path is not from the manifest (plus the fixed basemaps).

## When a PR changes model outputs

Update the matching `versions.main` (or `compare`) entries to the new bucket run folder / file names, rerun the generator, and include both files in the PR so the project on the branch matches that branch’s maps.
