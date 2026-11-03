# Arable cover (`ceh_landcover_arable`)

In-cell share of UKCEH LCM 2023 class Arable (code 3) on the model grid.

## What it shows

How much of each 100 m cell is that land-cover mapping (percent × 100; 0–10 000).

## What it does not show

Habitat condition, or cover in the surrounding landscape (use multiscale derivatives).

## Source and licence

**UKCEH Land Cover Map 2023** under UKCEH click-through / non-commercial terms. See [ukceh_lcm_2023](../sources/ukceh_lcm_2023.md).

## Pipeline

`ceh` rebuild step → merge as `ceh_landcover_arable`.

## Role

**Data** — land-cover predictor documentation.
