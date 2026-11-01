# Canopy cover share (VOM ≥2.5 m) (`vom_canopy_share`)

Fraction of each 100 m cell with EA VOM canopy-top height ≥ 2.5 m, from 1 m source tiles.

## What it shows

- Share of the cell occupied by classified VOM canopy (≥2.5 m)

## What it does not show

- Vegetation below 2.5 m
- Species or woodland inventory labels
- Leaf-on hedge structure

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: 1 m VOM download, Canopy mask at ≥2.5 m, Fraction of 100 m cell.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_canopy_share_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
