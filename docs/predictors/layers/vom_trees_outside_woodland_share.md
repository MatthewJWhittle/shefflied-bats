# Trees-outside-woodland share (`vom_trees_outside_woodland_share`)

Share of each 100 m cell with canopy that is not woodland-from-canopy (lone trees, small groups, and linear woody).

## What it shows

- Canopy outside woodland-from-canopy patches

## What it does not show

- Separate lone-tree vs hedge labels inside this share (see linear length)

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Canopy minus woodland-from-canopy, Fraction of 100 m cell.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_trees_outside_woodland_share_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
