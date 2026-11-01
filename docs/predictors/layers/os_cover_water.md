# Water feature cover (`os_cover_water`)

Share of each model cell covered by OS Vector Map District **Water** polygons.

## What it shows

In-cell water cover (%).

## What it does not show

Ecological quality of waterbodies, or streams missing from VMD.

## Source and licence

**OS Vector Map District**, Open Government Licence. Attribution: Contains OS
data © Crown copyright and database right.

## Pipeline

`os` rebuild step → `os-feature-cover.tif` → merge band `os_cover_water`.
Neighbourhood means (`os_cover_water_500m`, …) are derived later in the model
stack.

## Caveats

Coastal buffer cells that are open water dominate high cover values in study-area
QA — interpret inland water separately from sea/estuary buffer.

## Role

**Data** — OS cover predictor.
