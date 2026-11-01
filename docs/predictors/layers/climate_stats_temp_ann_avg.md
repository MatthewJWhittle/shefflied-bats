# Annual mean temperature (`climate_stats_temp_ann_avg`)

Long-term mean temperature from WorldClim 2.1 monthly averages.

## What it shows

Average of the twelve monthly mean-temperature surfaces (°C) after reprojection
to the 100 m model grid.

## What it does not show

Year-matched weather for bat records, or fine-scale microclimate.

## Source and licence

**WorldClim 2.1** (Fick & Hijmans 2017). Academic / non-commercial use; no
redistribution or commercial use without permission. Licence and access:
[worldclim_2_1](../sources/worldclim_2_1.md).

## Pipeline

`climate` rebuild step → monthly `tavg` → `calculate_climate_statistics` →
`temp_ann_avg` → merged band `climate_stats_temp_ann_avg`.

## Caveats

- 1970–2000 normal, not occurrence-year climate.
- Native ~1 km information content despite 100 m cells.
- Missing values along some coastal / outer edges of the climate grid.

## Role

**Data** — climate predictor documentation.
