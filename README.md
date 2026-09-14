# Sheffield Bats — HSM toolchain

Reusable **habitat suitability modelling (HSM)** pipeline. Sheffield / Yorkshire bats are the reference study; the same command-line path works for other species and areas.

This repo **trains and publishes** models. A separate app **maps** them.

| Piece | Role |
|-------|------|
| **This repo (`shefflied-bats`)** | Environmental layers → MaxEnt training → prediction rasters → versioned **model packages** |
| **[hsm-app](https://github.com/MatthewJWhittle/hsm-app)** | Online maps, species × activity layers, cite-able model cards |

They share **portable artefacts only** — not code. Each trained model is a small directory (`model.pkl` + `package.json`); map-ready surfaces are **Cloud Optimised GeoTIFFs (COGs)**. Contract: [docs/model-package-contract.md](docs/model-package-contract.md).

Everything runs through the **`sdm` command-line interface (CLI)**.

## Quick start

**You need:** Python 3.11+, Git, roughly 20 GB free for local data, and the external datasets listed under [Required data sources](#required-data-sources).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/MatthewJWhittle/shefflied-bats.git
cd shefflied-bats
uv sync
```

### Workflow

Paths default from `config.yml` (override per command as needed).

```bash
sdm setup          # study boundary
sdm data           # build and merge environmental layers
sdm background     # optional — training can create these on the fly
sdm train          # one model package per species × activity
sdm predict        # suitability surfaces (COG, project coordinate reference system)
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share --output-crs EPSG:3857 --cog   # Web Mercator COGs for map apps
```

Or run data → train → predict → visualise plots in one go:

```bash
sdm pipeline
```

Full command list: [sdm/README.md](sdm/README.md).

## What you get

**Model package** (per species × activity) under `data/sdm_models/{model_id}/`:

```
model.pkl
package.json   # schema_version 1, ordered feature_names, metrics, …
```

**Predictions** under `data/sdm_predictions/` — `all_predictions.tif` plus, by default, `prediction_{model_id}.tif` per model (project CRS from `config.yml`, usually British National Grid / EPSG:27700).

**hsm-app** expects tiled **EPSG:3857** COGs for both the project environmental stack and each suitability surface. Use `sdm export-rasters` before upload. Field mapping and upload notes: [docs/hsm-visualiser-integration.md](docs/hsm-visualiser-integration.md).

## Configuration

- `config.yml` — paths, CRS (default EPSG:27700), grid, MLflow
- `model_config.yml` / `variables_config.yml` — training and features
- Per-species overrides under `data/sdm_config/` and `data/sdm_tuning/`

```bash
sdm config
```

## Occurrence data

GeoJSON at `data/processed/bats-tidy.geojson` (or point `config.yml` with `sdm set-occurrence`). Expected columns:

| Column | Type | Notes |
|--------|------|-------|
| `unique_id` | string | Stable observation id |
| `latin_name` | string | e.g. `Pipistrellus pipistrellus` |
| `roosting` | string | `In Flight` or `Roosting` |
| `date` | string | `YYYY-MM-DD` |
| `x`, `y` | float | OSGB36 (EPSG:27700) |
| `accuracy` | float | Uncertainty in metres |
| `geometry` | Point | Location in EPSG:27700 |

## Layout

```
shefflied-bats/
├── config.yml
├── data/          # raw, processed, evs, models, predictions (mostly gitignored)
├── docs/          # contracts and integration guides
├── sdm/           # Python package + CLI
├── scripts/
├── notebooks/
└── tests/
```

## Required data sources

Large inputs are **not** in git. Download and place as below.

### OS Vector Map District

[OS Data Products](https://www.ordnancesurvey.co.uk/products/os-vectormap-district) — study-area tiles → `data/raw/big-files/os-vector-map`

### CEH Land Cover

[UKCEH Land Cover Maps](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) → `data/raw/big-files/CEH`

### BGS GeoCoast

[BGS GeoCoast Open](https://www.bgs.ac.uk/download/bgs-geocoast-open/) → `data/raw/big-files/BGS GeoCoast`

### Study boundary (Sheffield default)

ONS counties / unitary authorities GeoJSON for `sdm setup`:

`data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson`

Source: [ONS Geoportal — BDY_CTYUA](https://geoportal.statistics.gov.uk/search?q=BDY_CTYUA%202024&sort=Title%7Ctitle%7Casc)

## Development

```bash
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT — [LICENSE](LICENSE).
