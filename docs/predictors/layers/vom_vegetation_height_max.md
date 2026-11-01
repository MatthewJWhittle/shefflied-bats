# Vegetation canopy height — max (`vom_vegetation_height_max`)

Maximum top-of-canopy height from the EA **VOM** inside each 100 m cell.

## What it shows / does not show

Tallest classified canopy-top (m) in the cell. It does **not** show vegetation
under 2.5 m, understorey layering, or leaf-on hedge detail.

## Source

Same product as [`vom_vegetation_height_mean`](vom_vegetation_height_mean.md).
Official page:
[LIDAR Vegetation Object Model (VOM)](https://www.data.gov.uk/dataset/227ab487-e8f2-4cbb-b26a-9e6d3b662265/lidar-vegetation-object-model-vom)
— canopy-top height for objects **above 2.5 m**; National LiDAR flown
**leaf-off** in winter.

## Pipeline and caveats

Shares the `vom` rebuild path and 40 m pre-summary cap. Prefer the mean-layer
page for the full verified source wording; this page adds that **max** reacts
to single tall crowns or residual non-vegetation spikes.

## Role

**Data** — vegetation height predictor.
