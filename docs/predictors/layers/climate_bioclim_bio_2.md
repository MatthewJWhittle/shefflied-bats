# Mean diurnal temperature range (`climate_bioclim_bio_2`)

WorldClim **BIO2**: mean of monthly maximum − minimum temperature.

## What it shows

Typical diurnal temperature range (°C) under the 1970–2000 WorldClim normal.

## What it does not show

Record-year extremes or microclimate.

## Source and licence

WorldClim 2.1 bioclimatic surfaces — academic / non-commercial; no
redistribution without permission. See [worldclim_2_1](../sources/worldclim_2_1.md).

## Pipeline

`climate` step writes `bio.tif`; merge / canonicalize yields
`climate_bioclim_bio_2`. Selected into the model stack (`include=1`).

## Caveats

Shared with other WorldClim bands: temporal mismatch, coarse native grain,
possible coastal missing ring.

## Role

**Data** — climate predictor documentation.
