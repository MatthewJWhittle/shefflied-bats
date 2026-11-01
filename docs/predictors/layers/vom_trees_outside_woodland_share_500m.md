# Trees-outside-woodland share — 500 m neighbourhood

Focal mean of [`vom_trees_outside_woodland_share`](vom_trees_outside_woodland_share.md) over a 500 m (10-cell) window.

## Lineage

`kind: derived` → parent `vom_trees_outside_woodland_share`. Source VOM and canopy-geometry caveats live
on the parent page; this page does not copy them.

## Pipeline

Created in `generate_vom_canopy_features` via `add_multiscale_layers` /
`calculate_multiscale_variables` (windows 10 / 20 / 30 cells → 500 / 1000 / 1500 m).

## Role

**Data** — neighbourhood companion for the stage-one VOM canopy component.
