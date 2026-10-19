# Environmental layer pipeline — scaling assessment

Assessment for [#56](https://github.com/MatthewJWhittle/shefflied-bats/issues/56): where `sdm data` is slow or memory-bound today, and what has to change before England- and Great Britain–scale runs are workable. **Assessment only** — no pipeline behaviour change in the accompanying PR.

Related: library-reuse findings in [#51](https://github.com/MatthewJWhittle/shefflied-bats/pull/51) (do not repeat here; link only). Versioned layer store [#41](https://github.com/MatthewJWhittle/shefflied-bats/issues/41) is noted where storage format choices matter. Scaling goal tracks [#38](https://github.com/MatthewJWhittle/shefflied-bats/issues/38).

Raw numbers from the public-data profile runs: [`docs/artifacts/ev-scaling-profile-summary.json`](artifacts/ev-scaling-profile-summary.json). Re-run with:

```bash
uv run python scripts/profile_ev_scaling.py --sizes 10,50 -o data/scaling-bench
```

---

## Measured baselines

### Areas

| Label | Extent | Area (km²) | Role |
|-------|--------|------------|------|
| **10 km** | Axis-aligned square centred in the current study boundary (EPSG:27700) | 100 | Measured this assessment (public sources) |
| **50 km** | Same centre, larger square (~county scale) | 2 500 | Measured this assessment (public sources) |
| **Study** | `data/processed/boundary.geojson` | **14 369** | Provided full-rebuild timings (one machine, ~18 min) |
| **England** | Land area used for projection | 130 279 | Extrapolated |
| **Great Britain** | Land area used for projection | 209 331 | Extrapolated |

Area ratios used below: England / study ≈ **9.1×**, GB / study ≈ **14.6×**, England / 50 km ≈ **52×**, GB / 50 km ≈ **84×**, 50 km / 10 km = **25×**.

### Public-data profile (measured)

One Cloud Agent VM; sequential steps in one process. Peak RSS is process-tree high-water during each step, so later steps inherit retained allocations — treat **study-area provided peaks** as the better absolute memory ceiling, and use **10 km → 50 km growth** for how cost scales with area.

| Step | 10 km time (s) | 10 km peak (GB) | 50 km time (s) | 50 km peak (GB) | Time × (50/10) | Notes |
|------|----------------:|----------------:|----------------:|----------------:|---------------:|-------|
| `generate_terrain_data` | 14.7 | 0.66 | 68.2 | 1.51 | 4.6 | EA LiDAR WCS; tile count grows with AOI |
| `generate_terrain_stats` | 3.6 | 0.65 | 0.4 | 1.35 | (noisy) | Numba cold-start likely dominates 10 km |
| `generate_climate_data` (+ stats) | 16.9 | 1.82 | 16.7 | 2.46 | ≈1.0 | WorldClim **GBR national** tiles cached once |
| `generate_vom_data` | 12.6 | 1.68 | 69.3 | 2.39 | 5.5 | EA VOM WCS; same tiling pattern as terrain |
| `process_os_data` | 15.6 | 1.69 | 48.1 | 2.56 | 3.1 | Both AOIs used the same OS **SE** 100 km tile |
| `generate_coastal_distance` | 0.04 | 1.67 | 0.07 | 2.05 | — | **Synthetic** MHW polyline stand-in (not GB Boundary-Line) |
| `generate_ceh_lc_data` | 0.32 | 1.67 | 1.3 | 2.05 | 4.1 | **Synthetic** CEH-shaped GeoTIFF (licensed CEH unavailable) |
| `merge_ev_layers` | 1.2 | 1.67 | 2.7 | 2.37 | 2.2 | All inputs opened without dask chunks |
| `build_model_ev_stack` | 3.9 | 1.67 | 24.0 | 2.92 | 6.1 | Rolling neighbourhoods 10/20/30 cells |
| **Total wall** | **69** | — | **231** | — | 3.4 | Sub-linear vs 25× area (tile quantisation + national climate cache) |

Sources used: Environment Agency WCS (DTM/DSM/VOM), OS Vector Map District (open Downloads API), WorldClim 2.1 GBR ISO tiles, synthetic CEH + coast stand-ins of the same shape/type. No bat records, site names, or licensed CEH/GeoCoast.

### Provided study-area full rebuild (measured elsewhere)

| Step | Time (s) | Peak RSS (GB) |
|------|---------:|--------------:|
| `process_os_data` | 422 | 6.9 |
| `generate_vom_data` | 235 | 0.5 |
| `generate_terrain_data` | 92 | 0.5 |
| `generate_climate_data` | 52 | 2.5 |
| climate stats (within climate) | 17 | 3.0 |
| `generate_coastal_distance` | 32–52 | 0.9–1.3 |
| `generate_ceh_lc_data` | 1 | 0.5 |
| `merge_ev_layers` | 21 | 5.6 |
| Full run peak | — | **~10.3** |

A run on a box with ~10 GB free was **OOM-killed in merge/stack**. Memory is a first-class scaling limit.

---

## Step-by-step growth (from the code)

Entry point: `rebuild_environmental_layers` in `sdm/commands/data_preparation/rebuild_ev_layers.py` (CLI `sdm data`). Order matches `REBUILD_STEPS`.

### 1. Terrain — `generate_terrain_data`

- **What:** EA WCS GetCoverage for DTM + DSM at 10 m, tiled on a **fixed national grid** (`WCSDownloader.tile_bboxes_for_coverage` / `BoxTiler` origin `(0,0)`), merge, reproject to 100 m model grid, write GeoTIFF.
- **Time:** ≈ linear in AOI area (tile count ∝ area / tile footprint). Measured 10→50 km ≈ **4.6×** for 25× area (tile edges and concurrency damp the ratio).
- **Memory:** `open_mfdataset(...).load()` materialises the **whole AOI mosaic** at download resolution before reproject → **whole-raster-in-memory** for that AOI.
- **Download volume:** linear in tile count; no durable WCS tile cache today (temp files deleted after merge).

### 2. Terrain stats — `generate_terrain_stats`

- **What:** Slope/aspect/curvature/TWI-style Numba kernels on the DEM.
- **Time / memory:** linear in model-grid cells. Absolute times small vs downloads.

### 3. Climate — `generate_climate_data`

- **What:** Download WorldClim `GBR_wc2.1_30s_{var}.tif` into cache (`ClimateData.download_dataset`), open with `rioxarray` (**no chunks**), clip to AOI, reproject, write; optional annual stats.
- **Time:** download ≈ **O(1) national** once cached (measured 10 km ≈ 50 km). Clip/reproject grow with AOI cells.
- **Memory:** whole GBR GeoTIFF in memory during first cache write → explains ~2.5–3 GB peaks independent of small AOIs.
- **Download volume:** fixed per variable (national), then reuse.

### 4. CEH land cover — `generate_ceh_lc_data`

- **What:** Clip CEH LCM 10 m, dask `coarsen` class counts ([#46](https://github.com/MatthewJWhittle/shefflied-bats/pull/46)), aggregate habitats, reproject.
- **Time:** ≈ linear in AOI (measured synthetic 10→50 ≈ **4×**).
- **Memory:** peak tracks **chunk size**, not full national raster (already the right model). Still ends with an eager write of the AOI product.

### 5. Vegetation height — `generate_vom_data`

- **What:** Same WCS tiling pattern as terrain; `summarise_raster_metrics` coarsens 10 m → 100 m; reproject; write.
- **Time:** ≈ linear in tile count (measured **5.5×** for 25× area).
- **Memory:** whole-AOI load after tile merge (`.load()`), then summary — **whole-raster-in-memory** for the AOI.

### 6. Coastal distance — `generate_coastal_distance`

- **What:** Resolve coastline (Boundary-Line MHW / optional BGS / OSM), build target geometry, distance on a **coarser** grid (`resolution × 10`), reproject, clip.
- **Time:** grid build + STRtree ≈ O(cells × log features). Boundary-Line zip is **GB-wide once**.
- **Memory:** point grid for the AOI; coast geometry can be large for real Boundary-Line.
- **This profile:** synthetic one-line coast — times are **not** representative of GB MHW; use the provided study-area 32–52 s figures for absolute scale.

### 7. OS features — `process_os_data` (**slowest today**)

- **Download:** OS Vector Map District **100 km National Grid tiles** intersecting the buffered bbox (`osgb_100km_tiles_for_bbox`). Volume is **stepwise in 100 km tiles**, not smooth in km². Both profile squares hit only **SE**.
- **Load:** read all shapefiles in those tiles, then `sindex` filter to AOI.
- **Cover:** rasterise each theme at **10 m** over the full bbox (`calculate_feature_cover` → tempfile GeoTIFF → coarsen to 100 m). Intermediate pixels = 100× model cells → **worse than linear pressure** vs 100 m-only work; still O(area).
- **Distance:** `generate_point_grid` at 100 m (57 600 points @ 10 km buffered square → 409 600 @ 50 km) + Shapely **STRtree** nearest per theme — O(cells × log features). Exact geometry distance kept on purpose ([#51](https://github.com/MatthewJWhittle/shefflied-bats/pull/51) / #42); do not swap for EDT.
- **Study wall time 422 s / 6.9 GB** — dominates the rebuild.

### 8. Merge — `merge_ev_layers`

- **What:** For each input: `rxr.open_rasterio(..., masked=True)` **without chunks**, cast float32, reproject onto model grid, append; `xr.merge` all; `rio.clip`; `to_raster`.
- **Time:** ≈ linear in (bands × cells); measured growth mild on small squares.
- **Memory:** **every layer held at once** after the loop — study **5.6 GB**; OOM near 10 GB free. First single-machine breaker at England scale (see below).

### 9. Model stack — `build_model_ev_stack`

- **What:** Open merged stack whole; rename; layer clip; subset selected bands; `rolling` means for windows **10 / 20 / 30** cells (500 / 1000 / 1500 m); write `evs-to-model.tif`.
- **Time:** rolling is heavier than merge (measured **6×** from 10→50 km).
- **Memory:** whole multi-band array + neighbourhood temporaries — compounds merge pressure.

---

## Projected England and Great Britain

**Assumptions**

1. **Whole-array steps** (merge, stack, in-AOI WCS `.load()`, OS 10 m cover buffer): memory and CPU scale ≈ **linear in model cells** ≈ linear in area (ratio to study or to 50 km as noted).
2. **OS downloads:** scale with **count of 100 km tiles**, not km² (England ≫ one tile; study/profile squares often 1–few tiles).
3. **Climate download:** stays ~national GBR once cached; AOI work linear in cells.
4. **CEH:** keep dask coarsen; licensed national raster needed for real runs (stand-in used here).
5. **No tiling / streaming yet** — projections are “current code, larger AOI”.

| Step | Study (measured) | England (extrapolated) | GB (extrapolated) | Basis |
|------|------------------|------------------------|-------------------|--------|
| `process_os_data` | 422 s, 6.9 GB | **~1–1.5 h**, **≳40–60 GB** if still whole-AOI 10 m cover + full distance grid; **plus** tens of 100 km zip downloads | **~1.5–2.5 h** compute analog; more tiles | Time: area ×9–15 on compute after downloads; memory: linear cells; download: tile count |
| `generate_vom_data` | 235 s, 0.5 GB | **~35–45 min**, memory **~4–8 GB** per full mosaic load | **~55–70 min**, higher mosaic | Tile count × area; `.load()` AOI |
| `generate_terrain_data` | 92 s, 0.5 GB | **~15–20 min**, similar memory class to VOM | **~25–30 min** | Same WCS pattern ×2 layers |
| `generate_climate_data` | 52 s + 17 s stats; 2.5–3.0 GB | **~1–2 min** after cache; peak still **~3 GB** (national open) | same download; slightly larger clip | O(1) download |
| `generate_coastal_distance` | 32–52 s, 0.9–1.3 GB | **~5–10 min**, grid O(10⁷) cells | **~8–15 min** | Linear cells; GB MHW load once |
| `generate_ceh_lc_data` | 1 s, 0.5 GB | **tens of seconds–few min** if chunked | similar | Chunked path OK |
| `merge_ev_layers` | 21 s, **5.6 GB** | **~3–5 min**, **~45–55 GB** | **~5–8 min**, **~70–85 GB** | Linear bands×cells; **OOM on typical laptops** |
| `build_model_ev_stack` | (in full ~18 min budget) | grows faster than merge (rolling); **same memory class as merge+** | worse | Measured 6× time for 25× area |
| **Full peak RSS** | **~10.3 GB** | **≳50 GB** untiled | **≳80 GB** untiled | Merge/stack dominate |

**First step to break on a single machine at England scale:** `merge_ev_layers` / `build_model_ev_stack` holding the full multi-band grid (study already OOM near 10 GB free). OS 10 m cover + multi-million-point distance grids fail soon after or in parallel if merge is patched first without tiling.

---

## Options (adopt / try / skip)

Library micro-optimisations already ranked in [#51](https://github.com/MatthewJWhittle/shefflied-bats/pull/51) — reuse that list; below is **scale architecture** only.

### A. Raster vs vector-first (per layer)

| Layer / step | Recommendation | Gain / cost / risk / effort | Why |
|--------------|----------------|-----------------------------|-----|
| OS **distance** | **Adopt** keep vector STRtree → rasterise once | Gain: exact distances; Cost: low; Risk: low; Effort: none now | Already correct; [#51](https://github.com/MatthewJWhittle/shefflied-bats/pull/51) keeps STRtree over EDT |
| OS **cover** | **Try** vector-first / skip full 10 m GeoTIFF bounce | Gain: large CPU+disk at national scale; Cost: medium rewrite; Risk: medium (all_touched / fractions); Effort: medium | Hot path still tempfile 10 m raster ([#51] partial) |
| Coastal distance | **Adopt** vector coast → distance grid | Gain: already the model; Cost: Boundary-Line cache; Risk: low; Effort: low | Keep polyline STRtree path |
| CEH / VOM / terrain / climate | **Adopt** stay raster | — | Native gridded sources; CEH already dask-coarsened |
| Multiscale neighbourhoods | **Adopt** raster rolling/coarsen on tiled grids | Gain: matches model; Cost: halo logic; Risk: medium; Effort: medium | Needs halo (below), not a vector rewrite |

### B. National tiling + halos

| Choice | Recommendation | Detail |
|--------|----------------|--------|
| Fixed national tile scheme | **Adopt** | Align compute tiles to existing **OS 100 km** grid and EA WCS `BoxTiler` origin `(0,0)` so downloads and jobs share IDs. |
| Chunk / window size | **Try** | Start with **100 km** jobs; optionally subdivide to 25–50 km if peak RSS still high. |
| Windowed COG/Zarr reads | **Adopt** | Never `open_rasterio` the full national stack without `chunks=`. |
| **Halo / overlap** | **Adopt** | Multiscale max window **30 cells × 100 m = 1 500 m** → **≥1 500 m** (15 cells) halo beyond each tile’s interior before rolling (`center=True`). OS/coastal distances: include feature geometries (or coast) intersecting the **tile + max distance of interest** (clip distances already 0–7 000 m — use **7 000 m** feature halo for those layers). Terrain/VOM WCS tiles already national-aligned; re-merge with overlap matching kernel/filter support. **Check:** #38 closeness on one boundary-straddling tile vs untiled reference — grid shape and top-left corner match exactly; continuous layers pass when ≥95% of cells are within 1% of the layer’s range; categorical layers when ≥95% of cells match. |

### C. Dask / xarray vs per-tile multiprocessing

| Choice | Recommendation | Why |
|--------|----------------|-----|
| Dask-backed open for merge/stack/climate | **Adopt** | Today `compute()` / eager open too early (merge, climate cache write, WCS `.load()`). |
| Dask across whole England in one scheduler | **Skip** (for now) | Harder ops story; tile jobs are simpler and match OS 100 km downloads. |
| Multiprocessing / Batch **one tile per task** | **Adopt** | Embarrassingly parallel after halo rules; scale-to-zero friendly. |

### D. Merge and stack memory

| Choice | Recommendation | Why |
|--------|----------------|-----|
| Stream merge (write band-at-a-time or chunked Zarr/COG) | **Adopt** | Removes the 5.6 GB × area failure mode. |
| Hold all layers then `xr.merge` | **Skip** at national scale | Proven OOM path. |

### E. Download and caching

| Source | Recommendation | Why |
|--------|----------------|-----|
| OS 100 km zips | **Adopt** resumable on-disk cache keyed by tile ID (partially present) | England = many tiles; rate-limit politely. |
| EA WCS tiles | **Adopt** durable cache by `(coverage, tile, resolution)` | Temp-only today → re-download tax. |
| WorldClim GBR | **Adopt** keep national file cache | Already O(1) after first pull. |
| Boundary-Line | **Adopt** keep GB zip cache | One download. |

Faster libs for hotspots (GDAL proximity, xarray-spatial, numba): **skip** proximity/EDT for OS distances (accuracy); **try** only after tiling if a measured hotspot remains — see [#51](https://github.com/MatthewJWhittle/shefflied-bats/pull/51).

### F. Storage: COG vs Zarr (fits [#41](https://github.com/MatthewJWhittle/shefflied-bats/issues/41))

| Role | Recommendation | Why |
|------|----------------|-----|
| **Versioned layer store intermediates** (#41) | **Adopt Zarr** (chunked, one array per layer version) | Dedup unchanged layers; chunked write/read; natural for tile jobs; models can record Zarr group + layer version IDs. |
| **Model delivery / visualiser** (`evs-to-model`, predictions) | **Adopt COG GeoTIFF** | Already in toolchain (`rio-cogeo`); map clients expect COG; keep package contract. |
| Single national multi-band GeoTIFF as system of record | **Skip** | Fights versioning and memory. |

### G. Where it runs (design only — no cloud resources)

| Mode | Recommendation | Rough England cost (assumptions) |
|------|----------------|----------------------------------|
| Local workstation 32–64 GB | **Try** only after streaming merge | Still painful for OS/VOM downloads; overnight possible if tiled sequentially. |
| **Scale-to-zero GCP** — Cloud Run Jobs or Batch, one task per 100 km tile, outputs to Cloud Storage | **Adopt** for national runs | **Assumptions:** ~50 tiles × ~4 min (≈ our 50 km wall) × **4 vCPU / 16 GiB** Cloud Run Jobs (London/europe-west2 class rates ≈ **$0.000018 / vCPU-s** + **$0.000002 / GiB-s**). Compute ≈ 50 × 240 s × (4×1.8¢ + 16×0.2¢)/100000 ≈ **~$1–3 / full England rebuild** before free tier, plus **GCS** storage for caches/COGs (tens of GB) and any outbound reads of results. Inbound pulls from OS/EA into Google Cloud are not billed as network ingress. Spot **Batch** can cut compute further. **Local** electricity ≈ negligible; human time and laptop RAM dominate. |

---

## Ranked bottlenecks (national blockers)

1. **Merge / model stack whole-array memory** — study 5.6 GB merge, 10.3 GB full peak, OOM at ~10 GB free; England ≈ 50 GB untiled.
2. **`process_os_data` wall time + 10 m cover** — study 422 s; national tile fan-out + 10 m intermediates.
3. **VOM / terrain WCS download + eager `.load()`** — study 235 s + 92 s; linear tile growth, no durable cache.
4. **Multiscale rolling on full stack** — faster time growth than merge in 10→50 km profile (6×).
5. **Climate national open without chunks** — fixed ~3 GB; fine at AOI scale, still wasteful.

---

## Smallest ordered change set to reach England

What **stays the same:** band naming and clipping semantics; STRtree exact distances; CEH dask coarsen; WorldClim national cache idea; `sdm data` step registry; model package COG delivery.

| Order | Change | Prove it worked |
|------:|--------|-----------------|
| 1 | **Stream merge + stack** (chunked read/write; never hold all bands) | Peak RSS on 50 km and study rebuild **&lt; 8 GB** through merge/stack; #38 closeness vs current `all-evs.tif` / `evs-to-model.tif` |
| 2 | **Tile runner** on OS 100 km / WCS grid with **1 500 m** multiscale halo + **7 000 m** distance-feature halo | One boundary tile vs untiled reference passes #38 closeness (exact grid + 95% / 1%-of-range continuous, 95% match categorical); England job = N tile tasks |
| 3 | **Durable WCS + OS tile cache** | Second England (or 50 km) run skips re-download; bytes-from-network ≈ 0 for cached tiles |
| 4 | **OS cover without full 10 m tempfile** (in-memory rasterize / library path from #51) | `process_os_data` time on 50 km **≤ 0.5×** current profile; cover layer closeness check |
| 5 | **GCP Batch / Cloud Run Jobs per tile** (optional once 1–4 local) | England wall clock under chosen budget; cost within ~$1–5 compute at stated sizes |

---

## Implementation notes

- Deliverable for [#56](https://github.com/MatthewJWhittle/shefflied-bats/issues/56): this report plus `scripts/profile_ev_scaling.py` and `docs/artifacts/ev-scaling-profile-summary.json`. No `sdm/` pipeline behaviour change.
- CEH and coast in the profile runs are **synthetic stand-ins**; EA, OS, and WorldClim steps used live public services at respectful concurrency.
- Boundary-tile proof uses the #38 closeness check (not max absolute difference). GCP cost notes treat OS/EA pulls as unbilled ingress; storage and outbound reads remain the network-side risks.
- Matt: pick which ordered changes to schedule after review; only Matt merges optimisation work.
