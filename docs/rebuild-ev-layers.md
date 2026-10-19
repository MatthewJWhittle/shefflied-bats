# Rebuild environmental layers

## Command

```bash
sdm data --output-dir <folder>
```

Alias: `sdm data -o <folder>`.

### Flags

| Flag | Purpose |
|------|---------|
| `--output-dir` / `-o` | **Required.** Local folder for rebuilt layers (string, so `gs://` is not collapsed by pathlib). |
| `--cache-dir` | Download caches (WorldClim, OS tiles + parquets, Boundary-Line). Default: `<output-dir>/cache`. |
| `--fresh` | Ignore stale OS parquet and Boundary-Line caches; rebuild them from sources under `--cache-dir`. |
| `--skip-ceh` | Skip CEH land cover (manual GeoTIFF). |
| `--skip-coastal` | Skip coastal distance. |
| `--ceh-path` | Manual CEH LCM 2023 10 m GeoTIFF (default: git-ignored path under `data/raw/big-files/CEH/`). |
| `--no-live-download` | Offline mode for live OS / Boundary-Line downloads. |
| `--verbose` | Verbose logging. |

Examples:

```bash
sdm data --output-dir data/evs-rebuild
sdm data -o /tmp/evs-smoke --skip-ceh --skip-coastal
sdm data -o data/evs-rebuild --cache-dir /tmp/evs-cache --fresh
```

This is the single documented entry point that rebuilds every environmental layer from the public/live sources wired in the data-prep modules, then merges them to `<folder>/evs-to-model.tif`.

## Ordered steps

1. `generate_terrain_data` → `<folder>/terrain/`
2. `generate_terrain_stats` → `<folder>/terrain_stats.tif`
3. `generate_climate_data` → `<folder>/climate/` (WorldClim cache under `--cache-dir`/worldclim)
4. `generate_ceh_lc_data` → `<folder>/landcover/` (unless `--skip-ceh`)
5. `generate_vom_data` → `<folder>/vom/`
6. `generate_coastal_distance` → `<folder>/coastal_distance.tif` (unless `--skip-coastal`; Boundary-Line under `--cache-dir`/boundary-line)
7. `process_os_data` → `<folder>/os-feature-cover.tif`, `os-distance-to-feature.tif` (OS raw + parquets under `--cache-dir`)
8. `merge_ev_layers` → `<folder>/evs-to-model.tif`

## Write guards (refused destinations)

The command exits with code 2 and does not write if `--output-dir` is:

- the current model EV directory `data/evs` (or any path under it)
- any local path whose resolved parts include an `inputs` segment
- anything under `INPUT_DATA_ROOT` when that environment variable is set
- a `gs://…` URI (including pathlib-collapsed `gs:/…` and bucket `…/inputs` paths) — rebuild writes are local only

Choose a separate folder (for example `data/evs-rebuild` or a dated path) so the published / training layer set stays unchanged.

## Metrics

Each successful (or partially failed) run writes:

- `<folder>/rebuild_metrics.json`
- `<folder>/rebuild_metrics.csv`

Columns / fields: `step`, `duration_seconds`, `peak_rss_bytes` (per-step process-tree peak via sampling), `status`, `error`.

The JSON file also records `provenance`: the git commit SHA and each source's product/version or local file date (CEH path mtime when present).

## Using the rebuilt stack

```bash
sdm train --ev-file data/evs-rebuild/evs-to-model.tif
sdm predict --ev-path data/evs-rebuild/evs-to-model.tif
```

Or end-to-end after `sdm setup`:

```bash
sdm pipeline --output-dir data/evs-rebuild --fresh
```

## Implementation notes

Supports comparing a fresh rebuild against the live set without overwriting it. Layer comparison tooling and distance-geometry fixes are tracked separately. Use `--fresh` when source tiles change so stale OS parquets cannot hide updates.
