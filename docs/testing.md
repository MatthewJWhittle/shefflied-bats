# Testing

Tests guard behaviour the pipeline or a user relies on. A test should fail only when that behaviour breaks. Name the behaviour and who needs it; if you cannot, drop the test.

## What to write

1. **Known answers on tiny synthetics.** Prefer small rasters and point sets where the right answer is obvious by hand (cover, distance, density, focal means, edge handling). Pin the expected IDs or cell values; do not re-implement the code under test inside the assertion.
2. **Output contracts.** Layer and package checks cover CRS, grid alignment, dtype, nodata, band names, and value ranges. No NaN reaches the training table.
3. **Invariants.** Properties that must always hold — for example cover in 0–1, tiled output equals untiled with no seams (see [ADR 0002](adr/0002-uk-wide-scale-tile-by-tile.md)), and deterministic reruns with the same seed.
4. **Frozen regression area.** Compare a small fixed synthetic or public-only area within tolerance so refactors keep results. No site names, grid references, or record points in fixtures or assertions.
5. **Model sanity (sparingly).** A tiny fixture that fits end to end and scores above a floor; response curves keep a sensible ecological direction where that is claimed.
6. **Live sources.** Opt-in and separate from default CI (same pattern as today’s integration mark).

Many fast unit tests, fewer integration tests, very few end-to-end ones.

## What not to write

- Proof that deleted code is gone, or that a cleanup still “worked”
- Timing or peak-RAM races as pass/fail criteria
- Mirrors of private helpers or legacy paths that only restate the implementation
- Shape / “not equal” / smoke checks with no expected value or property
- Exact floats where a documented tolerance is right

## Review rule

For each test: what behaviour does it protect, and who relies on it?
