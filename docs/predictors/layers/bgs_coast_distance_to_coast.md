# Distance to coast (`bgs_coast_distance_to_coast`)

Coastal distance on the model grid. **Excluded** from `evs-to-model.tif`
selection (`include=0`); the `coastal` rebuild step may still write it for QA.

## What it shows

Metres to the configured coastline. Model-stack clipping uses 0–7000 m when the
band is present.

## What it does not show

Marine influence or habitat quality by itself — and beyond 7 km the clipped
layer is flat.

## Source

Default geometry from **OS Boundary-Line** (OGL). Historical band prefix
`bgs_coast_` retained. Optional BGS GeoCoast path is rebuild-only.

## Why excluded

Documented in [distance-to-coast.md](../modelling/distance-to-coast.md): heavy
saturation at the 7 km cap and redundant ecological signal.

## Role

**Modeller** — documented for lineage/QA; not a current model feature.
