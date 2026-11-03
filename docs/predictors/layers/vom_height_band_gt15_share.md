# Canopy height band >15 m (cell share) (`vom_height_band_gt15_share`)

Share of the 100 m cell with canopy-top height above 15 m.

## What it shows / does not show

Shows the quantity above. Does **not** show vegetation under 2.5 m, understorey
layering, woodland/TOW labels, or a hedge detector (#121 scope).

## Source

Same EA VOM product as [`vom_vegetation_height_mean`](vom_vegetation_height_mean.md).
Built by rebuild step `vom_bands` from 1 m tiles ([issue #121](https://github.com/MatthewJWhittle/shefflied-bats/issues/121)).

## Role

**Data** — woody vegetation structure for models that previously used the
correlated mean/std vegetation-height pair.
