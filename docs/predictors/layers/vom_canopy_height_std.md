# Canopy-only height standard deviation (`vom_canopy_height_std`)

Standard deviation of VOM canopy-top height within canopy cells of each 100 m cell (non-canopy excluded).

## What it shows

- Within-canopy height spread in the cell

## What it does not show

- Spread including non-canopy zeros/nodata

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Restrict to canopy ≥2.5 m, Std of heights in the 100 m cell, 40 m height cap applied on fine tiles before stats.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_canopy_height_std_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
