# AGENTS.md

Working notes for coding agents on this repo. Human setup and existing contributor guidelines: [CONTRIBUTING.md](CONTRIBUTING.md).

## What this project is

Habitat suitability modelling (HSM) toolchain for bats (Sheffield / Yorkshire is the reference run). The `sdm` CLI drives the path: rebuild environmental layers (`sdm data -o …`), optional background points, MaxEnt train, predict (COG rasters), and export for maps. Config lives in `config.yml`, `model_config.yml`, and `variables_config.yml`.

Layout: `sdm/` (package + CLI), `tests/`, `docs/` (contracts, rebuild, data layout), `data/` (mostly gitignored inputs and outputs), `infra/` (GCP bucket Terraform), `notebooks/`. The separate [hsm-app](https://github.com/MatthewJWhittle/hsm-app) visualiser consumes published packages and COGs; keep the boundary at artefact files (`model.pkl`, `package.json`, GeoTIFF COGs) — see [docs/model-package-contract.md](docs/model-package-contract.md). Prefer the CLI and config over one-off scripts ([CONTRIBUTING.md](CONTRIBUTING.md)).

## How to work

CI (`.github/workflows/python-tests.yml`) runs:

```bash
uv sync --all-groups
uv run pytest -m "not integration"
uvx flake8 sdm tests --count --select=E9,F63,F7,F82 --show-source --statistics
uvx flake8 sdm tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

PRs also run repo-hygiene and `terraform-validate` (`.github/workflows/terraform.yml`). Never skip or bypass CI. Keep PRs small — one change each. Stay in draft until reviews and CI agree. If you depart from an issue's brief, say so in the PR.

## Code practice

Before writing code, check whether a dependency we already use — or the standard library — already does it (e.g. xarray, dask, rioxarray, rasterio, geopandas, shapely, numpy, scipy, pandas, scikit-learn). Prefer the native call over hand-rolled loops, windowing, coordinate bookkeeping, or reimplemented algorithms. Hand-rolled code is only OK when it is clearly more correct for our data, or measurably faster or lighter, and the PR must say why. Replacements need an identity test plus a before/after table of time and peak memory. Do not add a heavy new dependency just to save a few lines. Reviews check every PR for this. Examples: [#46](https://github.com/MatthewJWhittle/shefflied-bats/pull/46) (~150 lines → dask chunks + `xarray.coarsen`; bitwise identical; ~3× faster); [#47](https://github.com/MatthewJWhittle/shefflied-bats/pull/47) (repo-wide scan).

Prefer a shared base with thin adapters over one-off patches. Bug fixes should state the root cause. Back any speed claim with a before/after measurement.

## Modelling practice

One modelling change at a time; write the reason first. Layer-level clipping in the environmental-layer build (for example distance layers 0–7000 m, curvature 5th–95th percentile) uses the whole map's values and never looks at the species records. Anything the model learns from the records — thresholds, caps, percentiles, or scaling fitted to training data — is fitted on training folds only. Still do not clip or cap inside the model pipeline. Adding or dropping a layer needs a stated ecological mechanism and scale, checks on held-out areas, and a per-layer table. Prefer the ecologically plausible model over a small accuracy gain.

## Data and safety

This repository is public. No secrets, site or roost names, grid references, or record points in git, PRs, issues, or docs. Licensed sources (CEH land cover map, BGS GeoCoast) live under the private bucket's `licensed-sources/` prefix and must never be committed. Rebuild output must not overwrite today's live layers (`data/evs`, paths under `inputs/`, or `INPUT_DATA_ROOT`); respect the existing `sdm data` write guards ([docs/rebuild-ev-layers.md](docs/rebuild-ev-layers.md)).

## Written work

In PRs, issues, commits, and docs, write under a role heading (e.g. Implementation notes). Never sign with, or refer to, bot or agent names. Never @-mention anyone except Matt when his decision is needed.
