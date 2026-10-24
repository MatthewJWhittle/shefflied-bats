# Library reuse assessment

Assessment for [#47](https://github.com/MatthewJWhittle/shefflied-bats/issues/47): where hand-rolled code reimplements something we already depend on (xarray, dask, rioxarray, rasterio, geopandas, shapely, scipy, numpy, pandas, scikit-learn, elapid, stdlib), and whether a swap is worth it.

**Objective** (from the #47 product review): swap only for significant line/file reduction **or** a measured speed/memory gain. If neither, keep. If hand-rolled was deliberate and still wins a benchmark, keep and say why.

**Timing:** this document is the first deliverable. Actual refactors wait until the #38 six-bug rebuild fixes are merged, and only for items Matt picks. This assessment must not delay rebuild, comparison, or retraining.

**Already done:** [#46](https://github.com/MatthewJWhittle/shefflied-bats/pull/46) (merged) replaced a hand-rolled CEH window walk with native dask chunks + `xarray.DataArray.coarsen`. That pattern is the reference for high-value swaps.

**Scope checked:** largest `sdm/` functions and files (top ~20 by line span), plus geospatial hot paths under `sdm/raster/`, `sdm/data/`, `sdm/occurrence/`, and modelling annotation/prediction. Thin CLI/config glue and correct library wrappers were skipped.

**How to read recommendations**

| Tag | Meaning |
|-----|---------|
| **replace** | Worth a follow-up PR after #38; identity test + before/after numbers required |
| **delete** | Unused / duplicate; delete with import cleanup (still a small PR + smoke) |
| **partial** | Keep the function; only swap an inner hot path |
| **keep** | Leave as-is; library alternative is worse, wrong, or not in-deps |

Measurements below use synthetic grids and point sets (no study-area data). Marked **measured** when timed in this assessment; otherwise **estimate**.

---

## Summary (ordered by value)

| # | Recommendation | Item | Why it ranks |
|---|----------------|------|--------------|
| — | done | CEH coarsen (#46) | Already shipped |
| 1 | replace | EV annotate via full GeoDataFrame | Large memory; live train/tune path |
| 2 | replace | Pairwise distance matrix for background filter | Measured ~20–80× slower + O(P×B) RAM |
| 3 | delete | Orphan `sdm/data/processing/core.py` APIs | ~200 LOC, no production callers |
| 4 | delete | Duplicate `sdm/data/ceh_processing.py` | Entire unused file |
| 5 | replace | Region density masks via `rio.clip` loop | Measured ~30× vs one `rasterize` |
| 6 | partial | Feature cover tempfile round-trip | Measured ~1.6× on small tile; more on large AOI |
| 7 | delete | Dead `aggregate_raster` (+ unused helpers) | Hand-rolled coarsen anti-pattern, unused |
| 8 | delete | Unused legacy extract / metrics / single-model apply | Maintenance only |
| 9 | keep | `mask_raster_to_boundary` | Hand path **faster** than `rasterio.mask.mask` |
| 10 | keep | Exact geometry distance (#42) | EDT faster but wrong (~25 m MAE on synthetic) |
| 11 | keep | Numba TWI / hydrology | No equivalent in current deps |
| 12 | keep | Geographic MaxEnt CV + density background | Domain semantics, already on elapid/scipy |

---

## 1. REPLACE — full EV stack → GeoDataFrame for point annotation

| | |
|---|---|
| **File / function** | `sdm/commands/modelling/train_sdm_models.py` — `convert_evs_to_geodataframe`, `annotate_points_with_evs` (used by `prepare_training_data`; also train + tune) |
| **Lines today** | 21 + 26 (call site in `prepare_training_data` ~20 more) |
| **Library replacement** | `elapid.annotate` (already wrapped as `sdm.data.processing.core.annotate_points`, used on another train path) |
| **Estimated lines after** | ~15–25 total (write/open stack once; annotate presence + background only) |
| **Expected performance** | **Measured (synthetic):** materialising ~50k cells × 30 vars → GeoDataFrame ≈ **25 MB** peak / 48 ms; ~200k cells × 30 ≈ **101 MB** / 308 ms. Full study stacks are larger → **estimate: hundreds of MB to multi‑GB** avoided by sampling at points only. CPU also better (no `sjoin_nearest` against the full grid). |
| **Risk** | Medium — `sjoin_nearest` tolerance vs elapid cell sampling / nodata may differ; regress presence and background EV values and CRS. `extract_grid_points` currently reuses the full conversion — give it a thin grid-only path if still needed. |
| **Recommendation** | **replace** |

---

## 2. REPLACE — pairwise distance matrix for background filtering

| | |
|---|---|
| **File / function** | `sdm/commands/modelling/train_sdm_models.py` — `get_distance_matrix`, `filter_background_by_distance` |
| **Lines today** | 18 + 31 |
| **Library replacement** | `scipy.spatial.cKDTree` on Point XY (or `sklearn.neighbors.NearestNeighbors`); keep only nearest distance, not the full matrix |
| **Estimated lines after** | ~12–18 for filter (drop matrix helper) |
| **Expected performance** | **Measured** (synthetic Point XY, projected): P=200/B=5k → 31 ms / 8.1 MB (shapely) vs 1.8 ms / 0.2 MB (cKDTree), **17×**; P=500/B=10k → **46×**; P=1k/B=20k → 635 ms / 160 MB vs 7.6 ms / 0.7 MB, **83×**. Distances matched within `1e-6`. Memory today is O(P×B). |
| **Risk** | Low for projected Point geometries. Do **not** confuse with #42 geometry-to-feature distance. |
| **Recommendation** | **replace** |

---

## 3. DELETE — orphan hand-rolled processing API

| | |
|---|---|
| **File / function** | `sdm/data/processing/core.py` — `merge_environmental_layers`, `process_occurrence_data`, `process_background_data`, `extract_environmental_data` (keep `annotate_points`) |
| **Lines today** | ~204 across the four functions; file 250 total |
| **Library replacement** | N/A for delete. If kept: merge → existing `merge_ev_layers` / `xr.merge`; background → `GeoSeries.sample_points`; extract → `rasterio.sample.sample_gen` or `elapid.annotate` |
| **Estimated lines after** | 0 for the four functions (+ export cleanup in `sdm/data/__init__.py` and `processing/__init__.py`) |
| **Expected performance** | Line/file reduction only (no live callers found outside re-exports). The extract loop would be slow if used. |
| **Risk** | Low if notebooks/external imports are checked first. |
| **Recommendation** | **delete** |

---

## 4. DELETE — duplicate CEH lookup module

| | |
|---|---|
| **File / function** | `sdm/data/ceh_processing.py` — `get_ceh_land_cover_codes_v2023`, `define_broad_habitat_categories` |
| **Lines today** | 83 (entire file) |
| **Library replacement** | Canonical copy already in `sdm/data/landcover.py` (used by CEH generation) |
| **Estimated lines after** | 0 |
| **Expected performance** | Line/file reduction; avoids divergent habitat maps |
| **Risk** | Low — zero importers found |
| **Recommendation** | **delete** |

---

## 5. REPLACE — per-region `rio.clip` density weighting

| | |
|---|---|
| **File / function** | `sdm/occurrence/sampling.py` — `weight_density_array_by_regions` |
| **Lines today** | 69 |
| **Library replacement** | One `rasterio.features.rasterize` (region-id grid) + numpy/xarray sums; keep `normalise_to_distribution` |
| **Estimated lines after** | ~30–40 |
| **Expected performance** | **Measured** (synthetic 200×200 grid): 5 regions → 22 ms clip vs 0.7 ms rasterize (**30×**); 20 → **39×**; 50 → 207 vs 6.6 ms (**31×**). |
| **Risk** | Medium — edge pixels / `all_touched` vs clip semantics; `tests/test_sampling.py` must stay green |
| **Recommendation** | **replace** |

---

## 6. PARTIAL — feature cover: drop tempfile GeoTIFF round-trip

| | |
|---|---|
| **File / function** | `sdm/raster/utils.py` — `rasterise_gdf`; `sdm/raster/processing.py` — `calculate_feature_cover` (inner tempfile → open → coarsen) |
| **Lines today** | 33 + ~40 contiguous I/O in cover path (cover function 118 total; coarsen already correct) |
| **Library replacement** | `rasterio.features.rasterize` (or keep `geometry_mask`) → in-memory `xarray` → existing `.coarsen(...).sum()` |
| **Estimated lines after** | ~25–40 for rasterise+coarsen (drop NamedTemporaryFile scaffolding) |
| **Expected performance** | **Measured (synthetic 500×500 @ 10 m → 100 m, 200 polygons):** tempfile write/read + coarsen **26 ms** vs in-memory **16 ms** (~1.6×). **Estimate:** larger AOIs see more I/O and peak-disk relief. |
| **Risk** | Medium — nodata, dtype, `all_touched`, CRS/transform must match OS cover tests |
| **Recommendation** | **partial** (keep cover math; remove disk bounce) |

---

## 7. DELETE — dead hand-rolled block-reduce

| | |
|---|---|
| **File / function** | `sdm/raster/utils.py` — `aggregate_raster` (optionally unused `tile_bounding_box` / `points_to_grid_squares` cluster if still unreferenced) |
| **Lines today** | 31 (`aggregate_raster`) |
| **Library replacement** | `DataArray.coarsen(...).sum()` (same as #46) or delete as unused |
| **Estimated lines after** | 0 |
| **Expected performance** | Line reduction; prevents reintroducing reshape-coarsen |
| **Risk** | Low (no callers) |
| **Recommendation** | **delete** |

---

## 8. DELETE — unused legacy modelling helpers

| | |
|---|---|
| **File / function** | `sdm/models/core/features.py` — `extract_features_from_raster` (46); `sdm/models/core/evaluation.py` — `evaluate_model_performance` / `evaluate_model_cv`; `sdm/models/maxent/maxent_model.py` — `apply_model_to_rasters` (124; predict path uses `apply_models_to_raster`) |
| **Lines today** | ~46 + ~90 + ~124 if all unused paths removed |
| **Library replacement** | extract → `rasterio.sample` / `elapid.annotate`; metrics → `sklearn.metrics`; single-model apply → existing elapid wrapper |
| **Estimated lines after** | 0 for dead symbols |
| **Expected performance** | Maintenance only |
| **Risk** | Low after confirming no notebook imports |
| **Recommendation** | **delete** (cleanup PR; not a perf project) |

---

## 9. KEEP — boundary mask (hand-rolled wins)

| | |
|---|---|
| **File / function** | `sdm/commands/modelling/predict_sdm_models.py` — `mask_raster_to_boundary` |
| **Lines today** | 89 |
| **Library replacement** | `rasterio.mask.mask(..., crop=False, filled=True)` |
| **Estimated lines after** | ~20–30 |
| **Expected performance** | **Measured (synthetic 8×400×400):** hand `geometry_mask` + band fill **1.9 ms** vs `rasterio.mask.mask` **7.2 ms** (~**4× faster** hand). Line savings exist but do **not** beat the measured speed. |
| **Risk** | N/A if kept. If thinned for LOC only, accept slower masking and re-check nodata / `all_touched` / band descriptions. |
| **Recommendation** | **keep** |

---

## 10. KEEP — exact distance to geometries (#42)

| | |
|---|---|
| **File / function** | `sdm/raster/processing.py` — `nearest_distances_strtree`, `calculate_distances`, `calculate_distance_to_geom` (OS + coastal callers) |
| **Lines today** | 31 + 73 + 43 |
| **Library replacement** | Tempting: rasterize features + `scipy.ndimage.distance_transform_edt` |
| **Estimated lines after** | N/A |
| **Expected performance** | **Measured (synthetic 40k cell centres, 3 lines):** STRtree **44 ms** vs EDT **1.5 ms**, but mean absolute error ≈ **25 m** on a 100 m grid — approximate, not exact planar distance to geometry. |
| **Risk** | High if swapped — changes EV layers; contradicts #42 |
| **Recommendation** | **keep** (already on Shapely `STRtree.query_nearest`) |

---

## 11. KEEP — Numba TWI / hydrology

| | |
|---|---|
| **File / function** | `sdm/data/terrain/stats.py` — pit fill, D8, flats, flow accumulation, `compute_twi_raster` (file 364 lines) |
| **Lines today** | ~300 of Numba kernels |
| **Library replacement** | None in current deps (`scipy.ndimage` / `np.gradient` cover slope only; RichDEM/pysheds would be new deps) |
| **Estimated lines after** | N/A |
| **Expected performance** | Domain correctness + Numba CPU; adding a hydro dependency is out of #47 scope |
| **Risk** | Very high if replaced |
| **Recommendation** | **keep** |

---

## 12. KEEP — geographic MaxEnt CV and density background core

| | |
|---|---|
| **File / function** | `cross_validate_maxent_model` / pipeline glue (`maxent_model.py`); `generate_background_points_from_data` and helpers (`occurrence/sampling.py`) |
| **Lines today** | Large orchestration already on elapid / sklearn / `np.histogram2d` / `gaussian_filter` |
| **Library replacement** | Plain `KFold` / generic samplers would change SDM behaviour |
| **Estimated lines after** | N/A |
| **Expected performance** | N/A — semantic keep |
| **Risk** | High (spatial leakage / bias recipe) |
| **Recommendation** | **keep** |

---

## Related: `apply_models_to_raster` DataFrame windows

| | |
|---|---|
| **File / function** | `sdm/models/maxent/maxent_model.py` — `apply_models_to_raster` (131) |
| **Lines today** | 131 |
| **Library replacement** | Prefer numpy / `elapid.geo.apply_model_to_array` where pipelines allow; window list can stay |
| **Estimated lines after** | ~40–70 if heavily thinned |
| **Expected performance** | **Measured (synthetic 128×128×40, sklearn Pipeline):** DataFrame predict **3.67 ms** vs ndarray **3.26 ms** (~12%); DataFrame construct alone ~0.02 ms. **Not** a clear win given named-column pipelines. Revisit only if profiling a real multi-model stack shows otherwise. |
| **Risk** | Medium–high — feature-name requirements |
| **Recommendation** | **keep** for now (optional profile later; delete unused sibling `apply_model_to_rasters` under item 8) |

---

## Done already — CEH dask coarsen (#46)

| | |
|---|---|
| **File / function** | `sdm/commands/data_preparation/environmental/generate_ceh_lc_data.py` — `_coarsen_class_areas_dask` / `coarsen_land_cover_categories` |
| **Status** | Merged on main (`c9e199a`). Default `coarsen_method="dask"`. Legacy `full` kept for identity/memory tests. |
| **Lesson** | Hand-rolled window walk → ~6 lines of `(da == code).coarsen(...).sum()` with lower peak RSS and faster runtime (see #46 PR numbers). |

---

## Explicit non-candidates (checked, skipped)

- Multiscale / summarise paths already on `xarray.coarsen` / `.rolling`
- `export_geotiff` / COG helpers — thin rasterio / rio-cogeo
- Boundary dissolve / simplify — already geopandas / topojson
- WCS terrain downloader — service protocol, not a library reinvention
- `grid_sample_points` — pandas groupby sample; no sklearn spatial twin
- Tune/explain CLI giants — Optuna/SHAP orchestration
- Duplicate thin landcover binary helpers — fold only if already editing that file

---

## Suggested follow-up order (after #38; Matt picks)

1. EV annotate → `elapid.annotate` (memory)
2. Background distance filter → `cKDTree` (CPU + memory)
3. Delete orphans (`processing/core` unused, `ceh_processing.py`, `aggregate_raster`, unused modelling helpers)
4. Region weight masks → `rasterize`
5. Feature-cover in-memory rasterize

Each chosen swap: identity (or layer-tolerance) test + before/after time and peak memory, as in #47.

---

## Implementation notes

- This file is the #47 assessment scan deliverable; no application logic changed in the accompanying docs-only PR.
- Swaps are intentionally deferred until the six rebuild bug fixes for #38 are merged.
- Matt: pick which replace/delete rows to schedule; keep rows need no action unless priorities change.
)
