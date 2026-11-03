# Suburban — 500 m neighbourhood (`ceh_landcover_suburban_500m`)

Focal mean of [`ceh_landcover_suburban`](ceh_landcover_suburban.md) over a 500 m (10-cell) window.

## Lineage

`kind: derived` → parent `ceh_landcover_suburban`. Source, licence, and classification caveats live on the parent page.

## Pipeline

`build_model_ev_stack` multiscale for the `ceh_landcover` group.

## Role

**Data** — multiscale land-cover predictor.
