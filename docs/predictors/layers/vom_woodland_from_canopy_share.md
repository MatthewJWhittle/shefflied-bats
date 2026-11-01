# Woodland-from-canopy share (`vom_woodland_from_canopy_share`)

Share of each 100 m cell classified as woodland from joined canopy that survives ~12 m inward erosion and covers ≥0.5 ha.

## What it shows

- Joined canopy blocks large enough to keep a core after ~10–15 m erosion

## What it does not show

- CEH woodland classes
- Managed woodland boundaries from maps

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Canopy mask, Binary erosion ~12 m, Keep components ≥0.5 ha intersecting a surviving core, Fraction of 100 m cell.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_woodland_from_canopy_share_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
