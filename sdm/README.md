# Species Distribution Modelling CLI

Command-line interface for environmental data preparation, model training, prediction, and export. Install from the repository root:

```bash
uv sync   # or: pip install -e .
```

## Commands

| Command | Purpose |
|---------|---------|
| `sdm setup` | Create the study boundary GeoJSON |
| `sdm data` | Generate and merge all environmental variable layers |
| `sdm background` | Generate background points for modelling |
| `sdm train` | Train MaxEnt models and write model packages |
| `sdm predict` | Apply trained models to the EV stack |
| `sdm export-rasters` | Reproject and/or COG-wrap rasters for sharing |
| `sdm visualize` | Partial dependence and related plots |
| `sdm explain` | SHAP interpretability plots |
| `sdm pipeline` | Run setup → data → train → predict → visualize |
| `sdm tune` | Hyperparameter search with Optuna |
| `sdm set-boundary` | Point `config.yml` at a boundary file |
| `sdm set-occurrence` | Point `config.yml` at occurrence GeoJSON |
| `sdm config` | Print current configuration |

Run `sdm COMMAND --help` for options.

## Example workflow

```bash
# Study area
sdm setup

# Environmental variables (terrain, climate, land cover, coastal, OS, merge)
sdm data

# Background points (optional)
sdm background \
  --occurrence-data-path data/processed/bats-tidy.geojson \
  --boundary-path data/processed/boundary.geojson

# Train
sdm train \
  --bats-file data/processed/bats-tidy.geojson \
  --ev-file data/evs/evs-to-model.tif

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

**`sdm predict`** writes `all_predictions.tif` and, by default, per-model `prediction_{model_id}.tif` files. Outputs use the project CRS (`config.yml`, typically EPSG:27700) and are COG-encoded unless `--no-cog` is passed. Use `--prediction-crs` to change the output CRS.

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

## Adding commands

1. Implement the command under `sdm/commands/`
2. Register it in `sdm/cli.py`
3. Document it in this file
