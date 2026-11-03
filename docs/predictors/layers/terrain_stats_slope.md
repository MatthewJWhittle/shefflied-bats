# Slope (`terrain_stats_slope`)

Local ground steepness derived from [`terrain_dtm`](terrain_dtm.md).

## What it shows

Slope angle on the model DEM. Rebuild default stores **radians**
(`generate_terrain_stats` → `output_slope_units="radians"`).

## What it does not show

Direction of slope (aspect), or canopy / building structure.

## Lineage

`kind: derived` from `terrain_dtm`. Read parent docs for DTM source, licence,
and elevation caveats — this page does not copy them.

## Pipeline

`terrain_stats` step: `process_dem_to_terrain_attributes` on the DTM GeoTIFF,
then merge as `terrain_stats_slope`.

## Caveats

- Same missing cells and coastal artefacts as the parent DTM.
- Confirm units before comparing to literature values published in degrees.

## Role

**Data** — derived terrain predictor.
