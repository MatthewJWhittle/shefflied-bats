# Broadleaved woodland cover (`ceh_landcover_broadleaved_woodland`)

In-cell share of UKCEH LCM 2023 **Broadleaved woodland** on the model grid.

## What it shows

How much of each 100 m cell is that LCM class. Values use percent × 100
(0–10 000), consistent with other `ceh_landcover_*` bands.

## What it does not show

Vegetation height, woodland quality, or cover in the surrounding landscape
(use multiscale `*_500m` / `*_1000m` / `*_1500m` bands for neighbourhood means).

## Source and licence

**UKCEH Land Cover Map 2023** 10 m raster under UKCEH click-through /
non-commercial terms. The source GeoTIFF is gitignored and must not be
committed. See [UKCEH LCM pages](https://www.ceh.ac.uk/data/ukceh-land-cover-maps).

## Pipeline

`ceh` rebuild step → class area layers → merge as `ceh_landcover_*`.

## Caveats

Map classes are not field-survey habitat polygons; garden trees and small
patches may fall into other classes.

## Role

**Data** — land-cover predictor documentation.
