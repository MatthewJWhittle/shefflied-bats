# Canopy edge length per hectare (`vom_canopy_edge_length_ha`)

Length of canopy / non-canopy edge (1 m edge pixels) expressed per hectare in each 100 m cell.

## What it shows

- Edge density of VOM canopy within the cell

## What it does not show

- Woodland-edge distance to a mapped polygon inventory

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Canopy edge pixels (4-connected), Count × 1 m / cell hectares.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_canopy_edge_length_ha_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
