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
sdm data -o data/evs-rebuild   # rebuild every EV layer into a caller-chosen folder
sdm background     # optional — training can create these on the fly
sdm train          # one model package per species × activity
sdm predict        # suitability surfaces (Cloud Optimised GeoTIFF / COG, project coordinate reference system / CRS)
sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/share --output-crs EPSG:3857 --cog   # Web Mercator COGs when sharing with map tools
```

Or run data → train → predict → visualise in one go (after `sdm setup`):

```bash
sdm pipeline -o data/evs-rebuild
```

### Rebuild environmental layers (`sdm data`)

One command rebuilds every public-source EV layer (terrain, climate, land cover, VOM, coastal, OS, merge) into an **output folder you choose**:

```bash
sdm data --output-dir data/evs-rebuild
sdm data -o data/evs-rebuild --cache-dir /tmp/evs-cache --fresh
# optional smoke skips (registry names):
sdm data -o /tmp/evs-smoke --skip ceh,coastal
```

Download caches (WorldClim, OS tiles/parquets, Boundary-Line) go under `--cache-dir` or, by default, `<output-dir>/cache`. Pass `--fresh` so stale OS parquets cannot hide source changes. Full flag list: [docs/rebuild-ev-layers.md](docs/rebuild-ev-layers.md).

**Write guards** — the command refuses destinations that would overwrite the live layer set:

| Refused | Example |
|---------|---------|
| Current EV directory | `data/evs` (and anything under it) |
| Any path with an `inputs` segment | `inputs/…`, `/mirror/inputs/data/evs` |
| Cloud URIs | `gs://…` (including bucket `…/inputs`; taken as a string so pathlib cannot collapse `gs://`) |
| Under `INPUT_DATA_ROOT` | local mirror of the published inputs tree |

Each run writes `rebuild_metrics.json` and `rebuild_metrics.csv` next to the layers (wall time and per-step peak RSS, plus git commit and source version/date provenance). Use `--ceh-path` when the CEH LCM GeoTIFF is not at the default git-ignored location. Point train/predict at the rebuilt stack explicitly, e.g. `--ev-file data/evs-rebuild/evs-to-model.tif`.

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

### Coastal distance (Boundary-Line)

**Live (default):** `sdm data` step 6 downloads OS [Boundary-Line](https://osdatahub.os.uk/data/downloads/open/BoundaryLine) mean high water polylines (`high_water_polyline`, CODE `0071`) from the [OS Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) (OpenData — no API key). One GB zip is cached under `data/raw/boundary-line/`.

**Manual optional (coastal-specific):** [BGS GeoCoast Open](https://www.bgs.ac.uk/download/bgs-geocoast-open/) → `data/raw/big-files/BGS GeoCoast` — pass explicitly via `coastline_path=` when needed.

Disable live download with `sdm data --no-live-download` when working fully offline (uses cached Boundary-Line or BGS manual drop if present).

### Study boundary (Sheffield default)

`sdm setup` downloads ONS counties / unitary authorities (BDY_CTYUA, December 2024 BFC) from the [Open Geography Portal](https://geoportal.statistics.gov.uk/) into `data/raw/ons-boundaries/` when no local file is present. No API key is required.

Manual fallback (May 2023 vintage still supported):

`data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson`

Disable live download with `sdm setup --no-live-download` when working fully offline.

## Cloud bucket layout

See [docs/data-layout.md](docs/data-layout.md) for every train/tune input and output path under `gs://habitat-suitability-modelling-data` (`inputs/` and `runs/`).

## Model-run output location

Training writes each run under `runs/<UTC date-time>-<short hex>/` beneath an output **root**. Default root: `paths.models` in `config.yml` (`data/sdm_models`). Override with the CLI `--output-dir` or:

```bash
export MODEL_OUTPUT_ROOT=gs://habitat-suitability-modelling-data
```

The bucket URI is the root; each train run creates `gs://habitat-suitability-modelling-data/runs/<timestamp-id>/`. Service accounts used for upload typically need permission to write only under the bucket’s top-level `runs/` prefix.

Remote roots use [fsspec](https://filesystem-spec.readthedocs.io/) / `gcsfs` with normal Google credentials (`gcloud auth application-default login` on a laptop, or `GOOGLE_APPLICATION_CREDENTIALS` on a server). Libraries that need local paths (pickle, GeoTIFF, and similar) write to a temporary directory first; the helper uploads the tree to the run folder. `model_results.csv` records final `gs://` paths for packages, not staging directories.

**Inputs from GCS (optional):** set a prefix for config-relative paths such as occurrence GeoJSON, boundary, and the environmental stack:

```bash
export INPUT_DATA_ROOT=gs://habitat-suitability-modelling-data/inputs
```

Vector inputs are fetched via fsspec; the environmental GeoTIFF is read through GDAL `/vsigs/` (requires GDAL Google Cloud access in the runtime environment).

**Inputs from a local mirror (recommended for large rasters on a laptop):** sync the bucket `inputs/` tree once (or when it changes), then point `INPUT_DATA_ROOT` at the local directory while keeping run outputs on GCS:

```bash
gcloud storage rsync gs://habitat-suitability-modelling-data/inputs ~/hsm-inputs --recursive
export INPUT_DATA_ROOT=~/hsm-inputs
export MODEL_OUTPUT_ROOT=gs://habitat-suitability-modelling-data
sdm train
```

Use `--dry-run` first to preview transfers. Add `--delete-unmatched-destination-objects` only if you want the local folder to match the bucket exactly (destroys local files that are not in the bucket prefix). Tilde in `INPUT_DATA_ROOT` is expanded to your home directory.

Browse or copy outputs with the gcloud CLI:

```bash
gcloud storage ls gs://habitat-suitability-modelling-data/runs/
gcloud storage cp -r gs://habitat-suitability-modelling-data/runs/20251015T120000Z-a1b2c3d4 ./local-copy
```

## Development

```bash
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT — [LICENSE](LICENSE).
