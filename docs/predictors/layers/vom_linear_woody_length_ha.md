# Linear woody length per hectare (`vom_linear_woody_length_ha`)

Skeleton length of narrow elongated non-woodland canopy (hedges / tree lines) per hectare in each 100 m cell.

## What it shows

- Approximate length density of narrow elongated canopy patches

## What it does not show

- OS or CEH hedge inventories
- Hedges below the 2.5 m VOM threshold

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Non-woodland canopy with narrow + elongated box-mean density windows (no labelling), Skeletonize, Length / cell hectares.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_linear_woody_length_ha_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
