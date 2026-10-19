# Agent and contributor principles

Human setup, CLI preference, visualiser boundary, and PR checks live in [CONTRIBUTING.md](CONTRIBUTING.md). This file states principles that apply to every change.

## Prefer what we already have

Before writing code, check whether a dependency we already use — or the standard library — already does it. Examples in this stack: xarray, dask, rioxarray, rasterio, geopandas, shapely, numpy, scipy, pandas, scikit-learn.

Prefer the native call over hand-rolled loops, windowing, coordinate bookkeeping, or reimplemented algorithms.

Hand-rolled code is only OK when it is clearly more correct for our data, or measurably faster or lighter — and the PR must say why. Any replacement of hand-rolled logic with a library call must prove the output is unchanged with an identity test, plus a before/after table of time and peak memory. Do not add a heavy new dependency just to save a few lines.

Reviews check every PR for this.

**Examples:** [#46](https://github.com/MatthewJWhittle/shefflied-bats/pull/46) (~150 lines replaced by dask chunks + `xarray.coarsen`; bitwise identical; ~3× faster). [#47](https://github.com/MatthewJWhittle/shefflied-bats/pull/47) is the repo-wide scan for the same pattern.

## Written work

Write from the role (for example Implementation notes). Never sign, name, or @-mention a bot. Tag only Matt if his decision is needed.
