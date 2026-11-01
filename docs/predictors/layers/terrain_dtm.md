# Digital terrain model (`terrain_dtm`)

Bare-earth elevation used as the primary terrain height predictor and as the
input DEM for terrain statistics.

## What it shows

Elevation of the ground surface in metres on the 100 m model grid
(EPSG:27700), derived from the Environment Agency LiDAR Composite DTM (1 m).

## What it does not show

- Heights of trees, buildings, or other surface objects (those are removed in
  the DTM product).
- A single survey date — the composite merges multi-year National / archive
  LiDAR.

## Source and licence

Environment Agency **LiDAR Composite DTM 1 m**, Open Government Licence v3.0.
Fetched live through the Defra/EA WCS endpoint during `sdm data` (`terrain`
step). Rasters are not stored in git.

## Pipeline

`generate_terrain_data` downloads DTM (and DSM) for the buffered study
boundary, then warps to the model grid. Band name after merge:
`terrain_dtm`.

## Caveats

- Sea / estuary cells inside a buffered boundary can show negative or near-zero
  elevations; inland negatives need separate checking (drained land vs artefact).
- Missing source tiles remain missing after reprojection.

## Role

**Data** — source description for modelling; no site-specific claims.
