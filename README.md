# Sheffield Bats — HSM toolchain

Habitat suitability modelling (HSM) for bats — **reference implementation** for a Yorkshire / Sheffield study, built as a reusable toolchain for any species and study area.

This repository **publishes** models: environmental variables, MaxEnt training, prediction COGs, and versioned **model packages** (`model.pkl` + `package.json`). A separate app **visualises** them for regional users (online maps, combined species × activity layers, cite-able model metadata).

Visualiser: **[hsm-app](https://github.com/MatthewJWhittle/hsm-app)**. The boundary between repos is **portable artefacts** (pickle + JSON + GeoTIFF COGs), not shared code. See [docs/model-package-contract.md](docs/model-package-contract.md).

## Quick start

### Prerequisites

- Python 3.11+
- Git
- ~20 GB disk space for local data and models
- Access to external datasets where required (OS, CEH, BGS) — see [Required data sources](#required-data-sources)

### Installation

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/MatthewJWhittle/shefflied-bats.git
cd shefflied-bats
uv sync
```

### Basic workflow

All steps use the **`sdm` CLI**. Paths default from `config.yml` and can be overridden per command.

```bash
# 1. Create study boundary
sdm setup

# 2. Generate and merge environmental variables
sdm data

# 3. Generate background points (optional — training can generate these on the fly)
sdm background

# 4. Train models → one package directory per species × activity
sdm train

# 5. Predict suitability surfaces (COG, project CRS)
sdm predict

# 6. Export Web Mercator COGs for map visualisers (when needed)
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share --output-crs EPSG:3857 --cog
```

Run the full pipeline in one go:

```bash
sdm pipeline
```

Command details: [sdm/README.md](sdm/README.md).

## Model packages and predictions

Training writes a **model package** per species × activity:

```
data/sdm_models/{model_id}/
├── model.pkl
└── package.json
```

Predictions land in `data/sdm_predictions/` as `all_predictions.tif` and, by default, per-model `prediction_{model_id}.tif` files.

For field definitions, CRS expectations, and EPSG:3857 export rules, see **[docs/model-package-contract.md](docs/model-package-contract.md)**.

To publish into **hsm-app** (HTTP API), see **[docs/hsm-visualiser-integration.md](docs/hsm-visualiser-integration.md)**.

## Configuration

- **`config.yml`** — paths, CRS (default EPSG:27700), spatial grid, MLflow settings
- **`model_config.yml`** / **`variables_config.yml`** — training and feature selection
- Per-species overrides under `data/sdm_config/` and `data/sdm_tuning/`

Inspect active settings:

```bash
sdm config
```

## Occurrence data

Prepare occurrence records as GeoJSON at `data/processed/bats-tidy.geojson` (or set another path with `sdm set-occurrence`). Expected columns:

| Column | Type | Notes |
|--------|------|-------|
| `unique_id` | string | Stable observation id |
| `latin_name` | string | e.g. `Pipistrellus pipistrellus` |
| `roosting` | string | `In Flight` or `Roosting` |
| `date` | string | `YYYY-MM-DD` |
| `x`, `y` | float | OSGB36 (EPSG:27700) |
| `accuracy` | float | Uncertainty in metres |
| `geometry` | Point | Location in EPSG:27700 |

## Project layout

```
shefflied-bats/
├── config.yml
├── data/                 # raw, processed, evs, models, predictions (mostly gitignored)
├── docs/                 # integration guides and contracts
├── sdm/                  # Python package and CLI
├── scripts/              # small helper utilities
├── notebooks/            # exploratory and publishing notebooks
└── tests/
```

## Required data sources

Large inputs are **not** committed. Download manually and place as below.

### OS Vector Map District

[OS Data Products](https://www.ordnancesurvey.co.uk/products/os-vectormap-district) — tiles for your study area:

`data/raw/big-files/os-vector-map`

### CEH Land Cover

[UKCEH Land Cover Maps](https://www.ceh.ac.uk/data/ukceh-land-cover-maps):

`data/raw/big-files/CEH`

### BGS GeoCoast

[BGS GeoCoast Open](https://www.bgs.ac.uk/download/bgs-geocoast-open/):

`data/raw/big-files/BGS GeoCoast`

### Study boundary (Sheffield default)

ONS boundary GeoJSON for county filtering in `sdm setup`:

`data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson`

Source: [ONS Geoportal — BDY_CTYUA](https://geoportal.statistics.gov.uk/search?q=BDY_CTYUA%202024&sort=Title%7Ctitle%7Casc)

## Development

```bash
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution notes.

## Licence

MIT — see [LICENSE](LICENSE).
