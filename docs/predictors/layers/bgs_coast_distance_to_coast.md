# Distance to coast (`bgs_coast_distance_to_coast`)

Coastal distance on the model grid. **Not used** as a model feature
(`include=0` in the EV CSV); the `coastal` rebuild step may still write it for QA.

## What it shows

Metres to the configured coastline. Model-stack clipping uses 0–7000 m when the
band is present.

## What it does not show

Marine influence or habitat quality by itself — and beyond 7 km the clipped
layer is flat.

## Source

Default geometry from **OS Boundary-Line** (OGL). See
[os_boundary_line](../sources/os_boundary_line.md). Historical band prefix
`bgs_coast_` retained. Optional BGS GeoCoast path is rebuild-only.

## Why not a model feature

Documented in [distance-to-coast.md](../modelling/distance-to-coast.md): heavy
saturation at the 7 km cap and redundant ecological signal.

## Role

**Modeller** — documented for lineage/QA; not a current model feature.
