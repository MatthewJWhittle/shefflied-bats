# Data-prep client status audit

Audit of environmental-variable and spatial data-prep clients in **sheffield-bats** (`sdm/commands/data_preparation/`, `sdm/data/`, `config.yml`, README). Read-only inventory for Building to discuss CLI data-prep work next.

**Audit date:** 2026-09-14  
**Scope:** Public/synthetic sources only; no SLR scenario wiring reviewed.

---

## Executive summary

| Finding | Detail |
|--------|--------|
| **CLI gap** | `sdm/README.md` and root `README.md` document granular commands (`sdm terrain`, `sdm landcover`, …) that are **not registered** in `sdm/cli.py`. Only **`sdm setup`** (boundary), **`sdm data`** (full EV pipeline), and **`sdm background`** expose data-prep today. |
| **Live vs manual** | **4 live clients** (terrain DTM/DSM, VOM, climate/WorldClim, boundary from ONS file). **3 manual big-files** (CEH land cover, BGS GeoCoast, OS Vector Map). |
| **Likely broken paths** | `sdm data` passes **directories** to `merge_ev_layers` for climate/landcover/vom; merge expects **file paths**. OS raw path is **`os-data`** in code vs **`os-vector-map`** in README. Climate **`run_stats=False`** in pipeline but `variables_config.yml` expects `climate_stats_*` bands. |
| **Orphans** | `ImageTileDownloader`, Sentinel/GEE helpers, legacy `merge_environmental_layers`, duplicate `ceh_processing.py` — not wired to CLI data-prep. |
| **tilearray** | Not present in this repo. Custom `WCSDownloader` (async tiled WCS 2.0.1) covers the same surface area as a future **tilearray** integration for EA LiDAR/VOM WCS (and potentially WMS/WMTS elsewhere). |

---

## CLI command map

| Documented CLI (`sdm/README.md`) | Registered in `sdm/cli.py` | Callable Python entry | Notes |
|----------------------------------|----------------------------|------------------------|-------|
| `sdm boundary` | **`sdm setup`** | `create_boundary` / `create_study_boundary_wrapper` | Name mismatch |
| `sdm terrain` | ❌ (use `sdm data`) | `generate_terrain_data` | |
| `sdm climate` | ❌ | `generate_climate_data` | |
| `sdm landcover` | ❌ | `generate_ceh_lc_data` | |
| `sdm vom` | ❌ | `generate_vom_data` | |
| `sdm terrain-stats` | ❌ | `generate_terrain_stats` | |
| `sdm coastal` | ❌ | `generate_coastal_distance` | |
| `sdm os` | ❌ | `process_os_data` | |
| `sdm merge` | ❌ | `merge_ev_layers` | |
| `sdm background` | ✅ | `generate_background_points_wrapper` | |
| `sdm data` | ✅ | Orchestrates all EV steps + merge | Monolithic pipeline |
| `sdm pipeline` | ✅ | setup → data → train → predict → visualize | |
| `sdm extract` | ❌ | — | Documented but not implemented |

---

## Client inventory

### Spatial / boundary

| Client | CLI | Data source | Implementation | Live API vs manual | Public API feasibility | Tests | Risks |
|--------|-----|-------------|----------------|--------------------|-------------------------|-------|-------|
| **Study boundary** | `sdm setup` | ONS BDY_CTYUA counties GeoJSON | **Implemented** — filters Yorkshire CTYUA23NM, dissolve, simplify | **Manual** — `data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) — FeatureServer/WFS/GeoJSON; no live client in repo yet (easy to wire) | `tests/test_boundary_simple.py` | Filename/column tied to May 2023 BFC product; docs say `sdm boundary` |
| **Background points** | `sdm background` | Bat occurrence GeoJSON + boundary | **Implemented** — density-smoothed sampling | N/A (derived) | N/A | `tests/test_generate_background_points.py`, `tests/test_sampling.py` | Depends on `paths.occurence_data` and boundary; not an external download client |

### Environmental variables (EV pipeline)

| Client | CLI (today) | Data source | Implementation | Live vs manual | Public API feasibility | Tests | Risks |
|--------|-------------|-------------|----------------|----------------|------------------------|-------|-------|
| **Terrain DTM/DSM** | `sdm data` step 1/7 | EA/Defra LiDAR composite **WCS 2.0.1** | **Implemented** — `WCSDownloader` + reproject/merge | **Live WCS** (no auth) | Already wired. Endpoints: [DTM WCS](https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs), [DSM WCS](https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs) | `tests/test_generate_terrain_data.py`, `tests/test_get_terrain_data.py`, `tests/test_ogc.py` (integration/slow) | Large AOI + 7 km buffer; concurrent tile limits; service uptime |
| **Terrain statistics** | `sdm data` step 2/7 | Derived from `evs/terrain/dtm_dsm_{res}m.tif` (DTM band) | **Implemented** — slope, aspect, TWI, TPI, roughness, etc. | N/A (local derivative) | N/A | `tests/test_terrain_stats.py`, `tests/test_data_terrain_stats.py` | Hard-coded input path in `sdm data`; band index assumption |
| **Climate (WorldClim 2.1)** | `sdm data` step 3/7 | `https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/GBR_wc2.1_30s_{var}.tif` | **Implemented** — `ClimateData` download + cache + clip/reproject | **Live HTTPS** (no auth) | Already wired; cache dir `data/raw/worldclim` | `tests/test_generate_climate_data.py` | **`run_stats=False` in pipeline** but models expect `climate_stats_*` in `variables_config.yml`; 30s (~1 km) source reprojected to 100 m |
| **CEH land cover (LCM 2023)** | `sdm data` step 4/7 | UKCEH Global Land Cover 2023 10 m GeoTIFF | **Implemented** processing (clip, coarsen, aggregate habitats) | **Manual** — default `data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif` | [UKCEH LCM](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) — EIDC GeoTIFF for modelling; WMS is view-only; **no AOI raster API**; non-commercial free / commercial licensed | `tests/test_generate_ceh_lc_data.py`, `tests/test_get_ceh_data.py` | **CEH licence** (research vs commercial); very large rasters; path is project-specific UUID folder |
| **VOM (vegetation height)** | `sdm data` step 5/7 | EA Vegetation Object Model **WCS 2.0.1** | **Implemented** — WCS download + `summarise_raster_metrics` | **Live WCS** (no auth) | Already wired: [VOM WCS](https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs) | `tests/test_generate_vom_data.py` (mock + `@pytest.mark.integration`) | Coverage extent/resolution limits; async tile download load |
| **Coastal distance** | `sdm data` step 6/7 | BGS GeoCoast **Authority Area Inundation** shapefile | **Implemented** — sea-zone polygon + distance raster | **Manual** — `data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp` | [BGS GeoCoast Open](https://www.bgs.ac.uk/datasets/geocoast-open/) + [MapServer](https://map.bgs.ac.uk/arcgis/rest/services/GeoCoast/GeoCoast_Open/MapServer) — REST live; confirm coastal in-scope | `tests/test_generate_coastal_distance.py` | Manual download; geometry processing sensitive to simplify/buffer params; **not an SLR scenario client** (uses inundation polygons for coastline geometry only) |
| **OS feature cover & distance** | `sdm data` step 7/7 | OS Vector Map District shapefiles | **Implemented** — parquet cache, road split, rasterise cover/distances | **Manual** — code uses `data/raw/big-files/os-data`; README/`load_os_shps` default is `os-vector-map` | [OS Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) / [Data Hub OpenData](https://osdatahub.os.uk/downloads/open/VectorMapDistrict) — OpenData automatable **without** API key; not wired yet | `tests/test_process_os_data.py` (component unit tests) | **OS licence**; **path mismatch** likely breaks fresh checkout; no end-to-end OS integration test |
| **Merge EV layers** | `sdm data` (final) | Prior step outputs | **Implemented** — reproject, merge, clip to boundary | N/A (orchestration) | N/A | `tests/test_merge_ev_layers.py` (mocked) | **`sdm data` passes dirs** (`evs/climate`, `evs/landcover`, `evs/vom`) but `load_and_preprocess_dataset` calls `rxr.open_rasterio(path)` on a **file**; likely fails for full pipeline |

### Utilities (not standalone CLI data-prep)

| Module | Wired to CLI? | Role | Status | Tests |
|--------|---------------|------|--------|-------|
| `split_raster_by_band` | Used by **`sdm predict`** only | Split multi-band COG/stack for per-model outputs | Implemented | None dedicated |
| `merge_environmental_layers` (`sdm/data/processing/core.py`) | ❌ | Legacy rasterio merge | Implemented, unused by current pipeline | None |
| `ImageTileDownloader` | ❌ | ArcGIS-style `exportImage` tiled REST downloader | Implemented library only | None |
| `Sentinel2CloudMasker` / `load_sentinel` | ❌ | GEE cloud masking; local `sentinel-2` mosaic loader | **Stub/orphan** — requires GEE credentials or manual files | None |
| `ceh_processing.py` | ❌ | Alternate CEH category maps | Duplicate of `landcover.py` aggregation logic | None |

---

## `sdm data` pipeline order (from `sdm/cli.py`)

```
1. generate_terrain_data      → data/evs/terrain/dtm_dsm_100m.tif
2. generate_terrain_stats     → data/evs/terrain_stats.tif
3. generate_climate_data      → data/evs/climate/{bio,tavg,prec,wind}.tif  (run_stats=False)
4. generate_ceh_lc_data       → data/evs/landcover/ceh-land-cover-100m.tif
5. generate_vom_data          → data/evs/vom/vom_summary_metrics_100m.tif
6. generate_coastal_distance  → data/evs/coastal_distance.tif
7. process_os_data            → data/evs/os-feature-cover.tif, os-distance-to-feature.tif
8. merge_ev_layers            → data/evs/evs-to-model.tif (config paths.ev_tiff)
```

Config defaults (`config.yml`): CRS **EPSG:27700**, model grid resolution **100 m**, study buffer **7000 m**.

---

## Manual files checklist (`data/raw/big-files/`)

| Dataset | Expected path (as in code / README) | Download |
|---------|--------------------------------------|----------|
| ONS counties boundary | `Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson` | [ONS Geoportal](https://geoportal.statistics.gov.uk/search?q=BDY_CTYUA%202024) |
| CEH LCM 2023 | `CEH/data/.../gblcm2023_10m.tif` (UUID folder varies) | [UKCEH LCM](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) |
| BGS GeoCoast | `BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp` | [BGS GeoCoast Open](https://www.bgs.ac.uk/download/bgs-geocoast-open/) |
| OS Vector Map District | **`os-data/`** (code) vs **`os-vector-map/`** (README) | [OS VMD product page](https://www.ordnancesurvey.co.uk/products/os-vectormap-district) |

Live caches (auto-created): `data/raw/worldclim/`, `data/processed/os-data/*.parquet`.

---

## tilearray overlap (future — not implemented)

| Current client | Protocol | tilearray opportunity |
|----------------|----------|------------------------|
| `WCSDownloader` (terrain, VOM) | WCS 2.0.1 GetCoverage, custom async tiling | Replace bespoke tiling with tilearray WCS backend; shared retry/cache policy |
| `ClimateData` | Direct GeoTIFF URL | Could stay as-is or use generic HTTP tile fetch |
| `ImageTileDownloader` | REST `exportImage` bbox tiles | Closest existing pattern to WMS-like tile arrays; candidate to consolidate under tilearray |
| CEH / OS / BGS | Manual bulk | WMS/WMTS/WCS from Defra/CEH/OS Data Hub **if** licences and endpoints allow — would need new clients |

---

## Test coverage summary

| Area | Test files | Integration / network |
|------|------------|------------------------|
| Terrain WCS | `test_generate_terrain_data.py`, `test_get_terrain_data.py`, `test_ogc.py` | Yes — `@pytest.mark.integration` / slow against EA WCS |
| VOM WCS | `test_generate_vom_data.py` | Yes — optional real WCS tests |
| Climate | `test_generate_climate_data.py` | Mostly unit; live download not CI-gated |
| CEH LC | `test_generate_ceh_lc_data.py`, `test_get_ceh_data.py` | Synthetic rasters |
| Coastal | `test_generate_coastal_distance.py` | Mocked + fixture shapefiles |
| OS | `test_process_os_data.py` | Synthetic geometries; no real OS tiles |
| Merge | `test_merge_ev_layers.py` | Mocked workflow |
| Boundary | `test_boundary_simple.py` | Mocked counties file |
| Background | `test_generate_background_points.py` | Synthetic occurrence data |
| Terrain stats algo | `test_terrain_stats.py`, `test_data_terrain_stats.py` | Pure numerical |

**Gap:** No single integration test runs `sdm data` end-to-end with real external services and manual big-files.

---

## Public API / licence notes (Researcher, Sep 2026)

Deepen the “Public API feasibility” column for the **manual** sources. MaxEnt / species distribution modelling needs **classified GeoTIFF (or equivalent raster) layers**, not painted Web Map Service (WMS) images.

### Already live (confirmed)

| Source | Access | Auth | Licence notes |
|--------|--------|------|---------------|
| EA terrain DTM/DSM + Vegetation Object Model | WCS 2.0.1 (as today); natural overlap with **tilearray** | None | Public Defra/EA spatialdata endpoints |
| WorldClim 2.1 | Direct HTTPS GeoTIFF zips / tiles (`geodata.ucdavis.edu`, [worldclim.org](https://worldclim.org/data/worldclim21.html)) | None | Citation required (Fick & Hijmans 2017); no fancy API |

### Manual sources — wiring cost

#### OS VectorMap District — **easy (wire first)**

- **Endpoint:** Ordnance Survey Downloads API — `https://api.os.uk/downloads/v1` (list products, then `/products/{id}/downloads`).
- **Auth:** **No API key** for OpenData product downloads. (API key / OAuth only for Premium *data packages*.)
- **Licence:** [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) via OS OpenData; acknowledge OS.
- **Docs:** [Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api); [product page](https://osdatahub.os.uk/downloads/open/VectorMapDistrict).
- **Note:** Audit row that implied an API key for automated OpenData access is **out of date** for this path.

#### ONS boundaries — **easy**

- **Endpoint:** [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) — ArcGIS FeatureServer / WFS / GeoJSON downloads (paginate large layers with `resultOffset` / `resultRecordCount`).
- **Auth:** None for public layers.
- **Licence:** Open Government Licence; datasets often also carry OS intellectual-property acknowledgment — follow the portal citation for the chosen vintage.
- **Use:** Replace the hard-coded May 2023 counties GeoJSON path with a live or version-pinned portal fetch.

#### BGS GeoCoast Open — **medium; confirm in-scope**

- **Endpoint:** ArcGIS REST MapServer — `https://map.bgs.ac.uk/arcgis/rest/services/GeoCoast/GeoCoast_Open/MapServer` (query / GeoJSON); bulk download from [BGS GeoCoast Open](https://www.bgs.ac.uk/datasets/geocoast-open/).
- **Auth:** None for open layers.
- **Licence:** Open Government Licence; acknowledge BGS © UKRI.
- **Caveat:** This is a **coastal** product (inundation / erosion domains). Confirm the inland Sheffield reference run still needs it before investing in a live client; the current coastal-distance step uses inundation polygons as a coastline proxy only.

#### UKCEH Land Cover Map — **hardest (keep EIDC / manual for now)**

- **What exists:** EIDC catalogue GeoTIFF downloads (e.g. [LCM2023 10 m GB](https://catalogue.ceh.ac.uk/documents/7727ce7d-531e-4d77-b756-5cc59ff016bd)); separate **WMS** for viewing (e.g. LCM2023 10 m WMS on EIDC).
- **Auth / licence:** Raster LCM free for **non-commercial** use via EIDC ([UKCEH LCM page](https://www.ceh.ac.uk/data/ukceh-land-cover-maps); [raster licence](https://eidc.ceh.ac.uk/licences/lcm-raster)). **Commercial** use needs a paid / bespoke licence. Vector land parcels are separately licensed (admin fee for non-commercial).
- **Why not WMS for modelling:** WMS returns painted map images. MaxEnt needs the **classified GeoTIFF** (or equivalent). There is **no clean “fetch AOI as classified raster” public API** — keep EIDC download (or order) unless a licensed path is agreed.
- **Wiring cost:** High relative to OS/ONS; licence is the blocker for any productised / commercial reuse.

### Suggested wiring order

1. **OS VectorMap** via Downloads API (no key) + fix `os-data` vs `os-vector-map` path drift.
2. **ONS boundaries** via Open Geography Portal FeatureServer / download.
3. Decide whether **GeoCoast** is in-scope for general toolchains (or coastal-only studies).
4. Leave **CEH land cover** on EIDC / manual until the licence story is clear.

Do **not** put these vendor details in the stranger-facing README — this audit is the right home.

---

## Recommended next steps (for Building discussion)

1. **Register granular Typer commands** (`terrain`, `climate`, `landcover`, …) matching `sdm/README.md`, or update docs to reflect `sdm data` only.
2. **Fix merge inputs** in `sdm data` — pass explicit `.tif` paths (or teach `merge_ev_layers` to expand directories / globs).
3. **Align OS raw path** (`os-data` vs `os-vector-map`) and document parquet cache behaviour.
4. **Enable `run_stats=True`** in climate step (or drop `climate_stats_*` from `variables_config.yml` until generated).
5. **Prioritise live API work** — OS Downloads + ONS portal first (see Public API notes above); keep EA WCS (terrain, VOM) + WorldClim; plan tilearray for shared WCS tiling; defer CEH automation pending licence.
6. **Defer / isolate** GEE Sentinel and `ImageTileDownloader` until a clear EV use-case exists.

---

## Source files reference

| Client | Command module | Data module |
|--------|----------------|-------------|
| Boundary | `commands/data_preparation/spatial/create_study_boundary.py` | `data/spatial/core.py` |
| Terrain | `commands/.../generate_terrain_data.py` | `data/terrain/core.py` (`WCSDownloader`) |
| Terrain stats | `commands/.../generate_terrain_stats.py` | `raster/terrain.py` |
| Climate | `commands/.../generate_climate_data.py` | `data/loaders/climate.py`, `data/climate.py` |
| Land cover | `commands/.../generate_ceh_lc_data.py` | `data/landcover.py` |
| VOM | `commands/.../generate_vom_data.py` | `WCSDownloader` (shared) |
| Coastal | `commands/.../generate_coastal_distance.py` | — |
| OS | `commands/.../process_os_data.py` | `data/os.py` |
| Merge | `commands/.../merge_ev_layers.py` | — |
| Background | `commands/.../generate_background_points.py` | `occurrence/sampling.py` |

CLI orchestration: `sdm/cli.py` (`setup`, `data`, `background`, `pipeline`).
