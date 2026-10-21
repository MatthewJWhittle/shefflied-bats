# Artificial light at night (ALAN) layer (#37)

Brown long-eared and other light-shy bats avoid lit areas. This layer adds yearly
VIIRS night-lights radiance to the environmental variables stack.

## Status

Scaffold on the current main pipeline. The rebuild step is registered as `alan`.
It **auto-skips** when the local VIIRS GeoTIFF is missing so existing rebuilds
keep working. No species `variables_config.yml` lists the band yet (off by
default). No radiance files are committed.

## Source (Matt to confirm)

| Candidate | Notes |
|-----------|--------|
| NASA Black Marble yearly VNP46 | Free; account/sign-up; confirm licence |
| Earth Observation Group (EOG) annual VIIRS | Free; account/sign-up; confirm licence |

Both measure upward glow at about **500 m**. They can miss a single lit lane
through woodland. Street-light point data is out of scope.

Place the chosen yearly GeoTIFF at:

```text
data/raw/big-files/alan/viirs_annual.tif
```

(or pass `alan_source_path` into the rebuild). Keep it in the bucket / local
cache only — **not in git**.

## How it is built

1. Load the local VIIRS GeoTIFF (no network download in code).
2. Reproject onto the model template grid (CRS, origin, 100 m cells, extent).
3. Upsample with **bilinear** resampling (document if you change this).
4. Write `<output-dir>/alan_viirs_radiance.tif` (band long name `viirs_radiance`;
   merge prefix `alan` → stack name `alan_viirs_radiance`).

```bash
# After the file is in place:
uv run sdm data --output-dir data/evs-rebuild --only alan
# Or full rebuild (alan runs when the source exists; otherwise skipped with a warning)
uv run sdm data --output-dir data/evs-rebuild
```

## Enabling for a species

1. Rebuild so `alan_viirs_radiance` is in `all-evs.tif` / `evs-to-model.tif`
   (add `alan_viirs_radiance` with `include=1` in `data/ev_clusters_selected.csv`
   once the band is produced).
2. Add `alan_viirs_radiance` to that species' `variables_config.yml`.
3. Leave other species unchanged.

## Done checks (from #37)

- Documented: source, licence, date, resolution, resampling.
- Grid/CRS/extent match the rest of the stack (covered by unit tests on a
  synthetic coarse raster).
- Plausibility map in the PR that first ships real VIIRS data: city centre
  brightest, moorland darkest.
- Brown long-eared roost run with the layer on vs latest #29 baseline (same
  seed, area folds): lit city-centre scores lower; woodland still highest;
  score no worse.

## Still needed from Matt

- Which product and year to use, and licence OK.
- Where the yearly file lives in the bucket.
- Go-ahead to flip `include=1` and to run the BLE comparison.
