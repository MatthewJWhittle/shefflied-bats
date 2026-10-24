# Artificial light at night (ALAN) layer (#37)

Brown long-eared and other light-shy bats avoid lit areas. This layer adds yearly
VIIRS night-lights radiance to the environmental variables stack.

## Status

Rebuild step `alan` on the current environmental-variables pipeline. It
**auto-skips** when the local VIIRS GeoTIFF is missing so existing rebuilds keep
working (the model stack then omits `alan_viirs_radiance` from the selected
subset). **Not enabled for any species yet** (brown long-eared / BLE wiring is
a separate second pass). No radiance files are committed.

## Source

| Field | Value |
|-------|--------|
| Product | EOG annual VNL **v2.2** (`average_masked`) |
| Year | **2025** composite (canonical; Matt, 24 Oct 2026) |
| Bucket (radiance) | `gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_annual.tif` |
| Bucket (coverage) | `gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_cf_cvg.tif` |
| Bucket (cf_cvg provenance) | `gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_cf_cvg.provenance.json` |
| Local cache (override) | `data/raw/big-files/alan/viirs_annual.tif` and `viirs_cf_cvg.tif` |
| Units | nW/cm²/sr |
| Native grid | EPSG:4326, ~15″ (~500 m) |
| Licence | EOG VNL CC-BY 4.0 (attribute Elvidge et al.; mark changes) |

Keep the global download in the bucket / local cache only — **not in git**.
Generation prefers the local cache when present, otherwise the bucket URIs above
(via GDAL `/vsigs/`). Explicit `--alan-source-path` / `--alan-cf-cvg-path`
(local or `gs://`) always win.

```bash
# Optional: sync bucket objects into the local cache
mkdir -p data/raw/big-files/alan
gsutil -m cp \
  gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_annual.tif \
  gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_cf_cvg.tif \
  gs://habitat-suitability-modelling-data/raw/big-files/alan/viirs_cf_cvg.provenance.json \
  data/raw/big-files/alan/
```

### Temporal mismatch

The 2025 annual composite does **not** match older occurrence years. Treat
lighting as a recent landscape context layer, not a year-matched covariate.
Document this whenever interpreting response curves or comparing to baselines
trained without ALAN.

## How it is built (ML-safe)

1. **Windowed read** of the study-area footprint (+ setup boundary buffer + a
   small bilinear interpolation margin). The global ~11.6 GB array is never
   loaded into `rio.reproject`.
2. **Preserve valid zeros** from `average_masked` (dark background). Never set
   nodata=0.
3. When `cf_cvg` is present, cells with coverage ≤ 0 become **NaN** (missing
   coverage). That is separate from radiance zeros.
4. **Bilinear** warp onto the exact model template grid (CRS, origin, 100 m
   cells, extent). Missing stays missing through the warp. 100 m cells still
   hold coarse satellite information.
5. Write `<output-dir>/alan_viirs_radiance.tif` with band name
   `alan_viirs_radiance` (same name in merge, selection CSV, and species
   configs). Provenance JSON records filename, year, product/version, checksum,
   units, validity treatment, and resampling.

```bash
# Scratch / dry-run only — never write live data/evs/
uv run sdm data --output-dir /path/to/scratch/evs-alan --only alan
```

### Raw radiance vs log1p

Default ML feature is **raw radiance**. The distribution is skewed; `log1p`
preserves zero and compresses extremes, but MaxEnt already standardises and
builds nonlinear features. Treat log1p as a **separate experiment** (same folds
as raw), not the default. Do not copy map percentile/colour stretches into the
feature.

## Enabling for a species (second pass)

Ingest and stack selection are in place (`include=1` in
`data/ev_clusters_selected.csv`). Species enablement is deferred:

1. Rebuild into a **scratch** output dir so `alan_viirs_radiance` is in
   `all-evs.tif` / `evs-to-model.tif`.
2. Add `alan_viirs_radiance` to that species' `variables_config.yml`
   (planned first candidate: *Plecotus auritus* roost — not in this PR).
3. Leave other species unchanged unless ecology supports lighting.

## Done checks (from #37)

- Documented: source, licence, year, resolution, resampling, temporal mismatch.
- Grid/CRS/extent match the rest of the stack (unit tests: valid zeros, cf_cvg
  missing mask, geographic→model reproject, merged band name).
- Plausibility: city centre brightest, moorland darkest (scratch rebuild).
- Species enablement / BLE roost on-vs-baseline comparison: **deferred** to a
  second pass (not part of this PR).
