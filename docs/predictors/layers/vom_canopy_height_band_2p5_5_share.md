# Canopy height band 2.5–5 m share (`vom_canopy_height_band_2p5_5_share`)

Fraction of each 100 m cell with VOM canopy height in [2.5, 5) m.

## What it shows

- Low canopy-top band share

## What it does not show

- Understorey below 2.5 m

## Recipe

EA VOM only (no CEH/OS). Rebuild with:

```bash
sdm data -o <scratch-or-local-path> --only vom_canopy
```

Processing: Height-band mask on canopy, Fraction of 100 m cell.

Neighbourhood means at 500 / 1000 / 1500 m reuse `calculate_multiscale_variables`
(see `vom_canopy_height_band_2p5_5_share_500m`).

## Caveats

Inherited VOM limits (2.5 m threshold, canopy-top only, leaf-off, mis-class risk)
plus canopy-geometry patch rules — see YAML `caveats`. Not selected into the live
EV model stack in this stage.

## Role

**Data / ecology** — stage-one VOM canopy component for later predictor recipes.
