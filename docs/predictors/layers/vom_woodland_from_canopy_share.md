# Woodland-from-canopy share (`vom_woodland_from_canopy_share`)

Share of each 100 m cell classified as woodland from joined canopy that keeps a near-solid ~25 m box-mean core, then filled through canopy by binary propagation (no EDT / labelling).

## What it shows

- Joined canopy reachable from a near-solid ~25 m box-mean core

## What it does not show

- CEH woodland classes
- Managed woodland boundaries from maps

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Canopy mask, near-solid box-mean core (~25 m), ``binary_propagation`` under canopy, Fraction of 100 m cell. No EDT / connected-component labelling.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_woodland_from_canopy_share_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
