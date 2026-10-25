# Species Distribution Modelling CLI

Command-line interface for environmental data preparation, model training, prediction, and export. Install from the repository root:

```bash
uv sync   # or: pip install -e .
```

## Commands

| Command | Purpose |
|---------|---------|
| `sdm setup` | Create the study boundary GeoJSON |
| `sdm data -o DIR` | Rebuild, merge (`all-evs.tif`), and build model stack (`evs-to-model.tif`); optional `--cache-dir` / `--fresh` / `--ceh-path` / `--plan` (refuses `data/evs`, `inputs/`, `gs://`) |
| `sdm background` | Generate background points for modelling |
| `sdm train` | Train MaxEnt models and write model packages |
| `sdm threshold` | Compare suitability cut-off rules on held-out CV scores; `--publish-run` writes METHOD.md cut-offs into the training run folder |
| `sdm predict` | Apply trained models to the EV stack |
| `sdm export-rasters` | Reproject and/or COG-wrap rasters for sharing |
| `sdm visualize` | Partial dependence and related plots |
| `sdm explain` | SHAP interpretability plots |
| `sdm pipeline -o DIR` | Run data → train → predict → visualize using the rebuilt EV folder |
| `sdm tune` | Hyperparameter search with Optuna |
| `sdm set-boundary` | Point `config.yml` at a boundary file |
| `sdm set-occurrence` | Point `config.yml` at occurrence GeoJSON |
| `sdm config` | Print current configuration |

Run `sdm COMMAND --help` for options.

## Example workflow

```bash
# Study area
sdm setup

# Environmental variables (terrain, climate, land cover, coastal, OS, merge, model stack)
# Writes into a separate folder so the live data/evs set is not overwritten.
sdm data --plan
sdm data --output-dir data/evs-rebuild

# Background points (optional)
sdm background \
  --occurrence-data-path data/processed/bats-tidy.geojson \
  --boundary-path data/processed/boundary.geojson

# Train
sdm train \
  --bats-file data/processed/bats-tidy.geojson \
  --ev-file data/evs/evs-to-model.tif

# Suitability thresholds (fit on training folds, score on held-out folds)
# Training writes thresholds.json/csv into the run folder (docs/thresholds/METHOD.md).
# Use --publish-run only when those files are missing (refuses overwrite); omit it for a comparison table.
sdm threshold \
  --models-dir data/sdm_models \
  --output-dir data/sdm_predictions/thresholds

# Predict (COG, project CRS from config.yml)
sdm predict \
  --ev-path data/evs/evs-to-model.tif \
  --models-dir data/sdm_models

# Web Mercator bundles for visualisers
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share --output-crs EPSG:3857 --cog

# Visualise
sdm visualize \
  --run-summary-path data/sdm_models/model_results.csv \
  --ev-raster-path data/evs/evs-to-model.tif
```

## Predict and export behaviour

**`sdm predict`** writes `all_predictions.tif` and, by default, per-model `prediction_{model_id}.tif` files. Outputs use the project CRS (`config.yml`, typically EPSG:27700) and are COG-encoded unless `--no-cog` is passed. Use `--prediction-crs` to change the output CRS. Pass `--publish` to also write versioned maps under `predictions/<model_id>/<YYYYMMDD>_<sha7>_cfg<hash>/` with names `prediction__<genus>_<species>_<behaviour>__<version>__suitability.tif`, `…__threshold_<rule>.tif`, and `…__run.json`. Publish root comes from `--publish-root`, else `PREDICTION_PUBLISH_ROOT`, else `prediction_publish.root` in `config.yml` (required when publishing). Existing version folders are never overwritten.

**`sdm export-rasters`** is for secondary bundles: pass explicit GeoTIFF paths, optionally `--output-crs EPSG:3857` and/or `--cog`. At least one of reprojection or COG encoding must be requested.

Model package layout and consumer expectations: [docs/model-package-contract.md](../docs/model-package-contract.md).

## Default paths

From `config.yml`:

| Setting | Default |
|---------|---------|
| Boundary | `data/processed/boundary.geojson` |
| Occurrence | `data/processed/bats-tidy.geojson` |
| Environmental stack | `data/evs/evs-to-model.tif` |
| Models | `data/sdm_models/` |
| Predictions | `data/sdm_predictions/` |

Shared MLflow tracking (Neon + GCS artefacts) is documented in
[docs/mlflow-tracking.md](../docs/mlflow-tracking.md).

## Adding commands

1. Implement the command under `sdm/commands/`
2. Register it in `sdm/cli.py`
3. Document it in this file
