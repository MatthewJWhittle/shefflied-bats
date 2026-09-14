# Sheffield Bats — HSM toolchain

Reusable **habitat suitability modelling (HSM)** pipeline driven by the **`sdm` command-line interface (CLI)**. The Sheffield bats study is a **reference run**; the same path works for other species and areas.

You get environmental layers, MaxEnt training, prediction rasters, and versioned **model packages** you can keep local or hand to whatever downstream tool you use. Package layout and raster expectations: [docs/model-package-contract.md](docs/model-package-contract.md).

## Quick start

**You need:** Python 3.11+, Git, roughly 20 GB free for local data, and the external datasets listed under [Required data sources](#required-data-sources) (what this reference run uses).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/MatthewJWhittle/shefflied-bats.git
cd shefflied-bats
uv sync
```

### Workflow

Paths default from `config.yml` (override per command as needed).

```bash
sdm setup          # study boundary (separate from pipeline)
sdm data           # build and merge environmental layers
sdm background     # optional — training can create these on the fly
sdm train          # one model package per species × activity
sdm predict        # suitability surfaces (Cloud Optimised GeoTIFF / COG, project coordinate reference system / CRS)
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share --output-crs EPSG:3857 --cog   # Web Mercator COGs when sharing with map tools
```

Or run data → train → predict → visualise in one go (after `sdm setup`):

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

**Predictions** under `data/sdm_predictions/` — `all_predictions.tif` plus, by default, `prediction_{model_id}.tif` per model (project CRS from `config.yml`, usually British National Grid / EPSG:27700). `sdm predict` emits COGs in the project CRS by default.

**Sharing:** when a consumer needs **Web Mercator (EPSG:3857)** tiled COGs — for example a web map or HTTP upload workflow — use `sdm export-rasters` on prediction rasters (and, if needed, the environmental stack). Package layout, band naming, and CRS rules: [docs/model-package-contract.md](docs/model-package-contract.md). Optional HTTP upload and visualiser notes: [docs/hsm-visualiser-integration.md](docs/hsm-visualiser-integration.md).

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

Large inputs are **not** in git. Download and place as below for the Sheffield / Yorkshire reference run.

### OS Vector Map District

**Live (preferred):** `sdm data` step 7 downloads missing 100 km tiles from the [OS Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) (OpenData — no API key required). Shapefiles are cached under `data/raw/big-files/os-vector-map/<TILE>/`.

**Manual fallback:** [OS Vector Map District](https://www.ordnancesurvey.co.uk/products/os-vectormap-district) shapefiles → `data/raw/big-files/os-vector-map/`

Optional API key for OS Data Hub premium packages (not required for OpenData):

```bash
export OS_DATA_HUB_KEY=your-key-here
```

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
