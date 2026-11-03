# Broadleaved woodland — 500 m neighbourhood

Focal mean of
[`ceh_landcover_broadleaved_woodland`](ceh_landcover_broadleaved_woodland.md)
over a 500 m (10-cell) window.

## Lineage

`kind: derived` → parent `ceh_landcover_broadleaved_woodland`. Source, licence,
and classification caveats live on the parent page; this page does not copy them.

## What it adds

Landscape context: woodland share in the surrounding 500 m, not only the cell.

## Pipeline

Created in `build_model_ev_stack` via `calculate_multiscale_variables` for the
`ceh_landcover` group (windows 10 / 20 / 30 cells → 500 / 1000 / 1500 m).

## Role

**Data** — worked example of multiscale lineage for the first documentation batch.
