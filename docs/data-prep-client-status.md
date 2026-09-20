# Data-prep client status audit

Audit of environmental-variable and spatial data-prep clients in **sheffield-bats** (`sdm/commands/data_preparation/`, `sdm/data/`, `config.yml`, README). Read-only inventory for Building to discuss CLI data-prep work next.

**Audit date:** 2026-09-14  
**Scope:** Public/synthetic sources only; no SLR scenario wiring reviewed.

---

## Executive summary

| Finding | Detail |
|--------|--------|
| **CLI gap** | **Resolved (docs-match-CLI).** Root `README.md` and `sdm/README.md` document only registered commands (`sdm setup`, `sdm data`, `sdm background`, modelling commands). Granular Python entry points remain callable from code/notebooks but are not separate Typer commands. |
| **Live vs manual** | **7 live clients** (terrain DTM/DSM, VOM, climate/WorldClim, **ONS boundaries**, **OS Vector Map District**, **OS Boundary-Line coastline**). **2 manual big-files** (CEH land cover, BGS GeoCoast optional). ONS + OS VMD + Boundary-Line keep manual drops as fallback. |
| **Likely broken paths** | **Fixed:** `sdm data` now passes explicit `.tif` paths via `build_ev_dataset_inputs`. OS raw path aligned to **`os-vector-map`** (README + `load_os_shps`); parquet cache stays **`data/processed/os-data`**. **Open:** climate **`run_stats=False`** in pipeline but `variables_config.yml` expects `climate_stats_*` bands. |
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
| **Study boundary** | `sdm setup` | ONS BDY_CTYUA counties GeoJSON | **Implemented** — filters Yorkshire CTYUA names, dissolve, simplify | **Live** — FeatureServer fetch to `data/raw/ons-boundaries/`; **manual fallback** — May 2023 GeoJSON under `data/raw/big-files/` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) — December 2024 BFC FeatureServer (no auth) | `tests/test_boundary_simple.py`, `tests/test_ons_download.py` | CTYUA column varies by vintage (`CTYUA24NM` vs `CTYUA23NM`); legacy manual path still honoured |
| **Background points** | `sdm background` | Bat occurrence GeoJSON + boundary | **Implemented** — density-smoothed sampling | N/A (derived) | N/A | `tests/test_generate_background_points.py`, `tests/test_sampling.py` | Depends on `paths.occurence_data` and boundary; not an external download client |

### Environmental variables (EV pipeline)

| Client | CLI (today) | Data source | Implementation | Live vs manual | Public API feasibility | Tests | Risks |
|--------|-------------|-------------|----------------|----------------|------------------------|-------|-------|
| **Terrain DTM/DSM** | `sdm data` step 1/7 | EA/Defra LiDAR composite **WCS 2.0.1** | **Implemented** — `WCSDownloader` + reproject/merge | **Live WCS** (no auth) | Already wired. Endpoints: [DTM WCS](https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs), [DSM WCS](https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs) | `tests/test_generate_terrain_data.py`, `tests/test_get_terrain_data.py`, `tests/test_ogc.py` (integration/slow) | Large AOI + 7 km buffer; concurrent tile limits; service uptime |
| **Terrain statistics** | `sdm data` step 2/7 | Derived from `evs/terrain/dtm_dsm_{res}m.tif` (DTM band) | **Implemented** — slope, aspect, TWI, TPI, roughness, etc. | N/A (local derivative) | N/A | `tests/test_terrain_stats.py`, `tests/test_data_terrain_stats.py` | Hard-coded input path in `sdm data`; band index assumption |
| **Climate (WorldClim 2.1)** | `sdm data` step 3/7 | `https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/GBR_wc2.1_30s_{var}.tif` | **Implemented** — `ClimateData` download + cache + clip/reproject | **Live HTTPS** (no auth) | Already wired; cache dir `data/raw/worldclim` | `tests/test_generate_climate_data.py` | **`run_stats=False` in pipeline** but models expect `climate_stats_*` in `variables_config.yml`; 30s (~1 km) source reprojected to 100 m |
| **CEH land cover (LCM 2023)** | `sdm data` step 4/7 | UKCEH Global Land Cover 2023 10 m GeoTIFF | **Implemented** processing (clip, coarsen, aggregate habitats) | **Manual** — default `data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif` | [UKCEH LCM](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) — EIDC GeoTIFF for modelling; WMS is view-only; **no AOI raster API**; non-commercial free / commercial licensed | `tests/test_generate_ceh_lc_data.py`, `tests/test_get_ceh_data.py` | **CEH licence** (research vs commercial); very large rasters; path is project-specific UUID folder |
| **VOM (vegetation height)** | `sdm data` step 5/7 | EA Vegetation Object Model **WCS 2.0.1** | **Implemented** — WCS download + `summarise_raster_metrics` | **Live WCS** (no auth) | Already wired: [VOM WCS](https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs) | `tests/test_generate_vom_data.py` (mock + `@pytest.mark.integration`) | Coverage extent/resolution limits; async tile download load |
| **Coastal distance** | `sdm data` step 6/7 | **OS Boundary-Line** `high_water_polyline` (CODE `0071` MHW) | **Implemented** — MHW polyline distance raster (Britain-wide) | **Live + manual fallback** — OS Downloads API → `data/raw/boundary-line/`; optional BGS GeoCoast manual; OSM Overpass bbox fallback on download failure | [OS Boundary-Line OpenData](https://osdatahub.os.uk/data/downloads/open/BoundaryLine) via [Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) (no auth); [BGS GeoCoast](https://www.bgs.ac.uk/datasets/geocoast-open/) optional manual | `tests/test_generate_coastal_distance.py`, `tests/test_coastline_download.py` | GB zip ~700 MB; full GB MHW used for inland AOIs; BGS sea-zone path retained for explicit manual override; **not an SLR scenario client** |
| **OS feature cover & distance** | `sdm data` step 7/7 | OS Vector Map District shapefiles | **Implemented** — live download via `sdm/data/os_download.py`, parquet cache, road split, rasterise cover/distances | **Live + manual fallback** — OS Downloads API → `data/raw/big-files/os-vector-map/<TILE>/`; parquet cache `data/processed/os-data` | [OS Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) — OpenData **without** API key; optional `OS_DATA_HUB_KEY` for premium packages | `tests/test_process_os_data.py`, `tests/test_os_download.py` (mocked HTTP + tile resolution) | **OS licence**; Skipton AOI (~3 km) spans tiles **SD+SE** (~190 MB zipped) |
| **Merge EV layers** | `sdm data` (final) | Prior step outputs | **Implemented** — reproject, merge, clip to boundary | N/A (orchestration) | N/A | `tests/test_merge_ev_layers.py` (mocked + `build_ev_dataset_inputs`) | **`sdm data` uses `build_ev_dataset_inputs`** — explicit `.tif` paths per layer |

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
| ONS counties boundary | Live cache `data/raw/ons-boundaries/counties_unitary_authorities_dec2024_bfc.geojson`; manual fallback `data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson` | [ONS Geoportal — BDY_CTYUA Dec 2024](https://geoportal.statistics.gov.uk/) |
| CEH LCM 2023 | `CEH/data/.../gblcm2023_10m.tif` (UUID folder varies) | [UKCEH LCM](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) |
| OS Boundary-Line MHW | Live cache `data/raw/boundary-line/high_water_polyline.shp` | [OS Boundary-Line OpenData](https://osdatahub.os.uk/data/downloads/open/BoundaryLine) |
| BGS GeoCoast (optional) | `BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp` | [BGS GeoCoast Open](https://www.bgs.ac.uk/download/bgs-geocoast-open/) |
| OS Vector Map District | **Live:** OS Downloads API → **`os-vector-map/<TILE>/`**; **manual:** flat `os-vector-map/`; parquet cache **`data/processed/os-data/`** | [OS VMD product page](https://www.ordnancesurvey.co.uk/products/os-vectormap-district) / [Downloads API](https://docs.os.uk/os-apis/accessing-os-apis/os-downloads-api) |

Live caches (auto-created): `data/raw/worldclim/`, `data/raw/boundary-line/`, `data/processed/os-data/*.parquet`.

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
| Coastal | `test_generate_coastal_distance.py`, `test_coastline_download.py` | Mocked Downloads API + fixture shapefiles; Whitby smoke artifact |
| OS | `test_process_os_data.py`, `test_os_download.py` | Mocked Downloads API; Skipton tile-resolution smoke |
| Merge | `test_merge_ev_layers.py` | Mocked workflow |
| Boundary | `test_boundary_simple.py`, `test_ons_download.py` | Mocked + optional live ONS smoke |
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

#### OS Boundary-Line coastline — **wired (live default)**

- **Endpoint:** OS Downloads API — `https://api.os.uk/downloads/v1/products/BoundaryLine/downloads?area=GB&format=ESRI%C2%AE%20Shapefile` (one GB zip, not tile soup).
- **Layer:** `high_water_polyline`; filter `CODE` `0071` (mean high water / springs).
- **Auth:** None for OpenData.
- **Licence:** Open Government Licence; acknowledge OS.
- **Cache:** `data/raw/boundary-line/` (extracts MHW layer only).
- **Fallback:** OSM Overpass coastline (bbox-scoped) if Boundary-Line download fails; BGS GeoCoast manual optional for coastal-specific jobs.
- **Smoke artifact:** `docs/artifacts/whitby-coastline-smoke.json`.

#### BGS GeoCoast Open — **manual optional (coastal-specific)**

- **Endpoint:** ArcGIS REST MapServer — `https://map.bgs.ac.uk/arcgis/rest/services/GeoCoast/GeoCoast_Open/MapServer` (query / GeoJSON); bulk download from [BGS GeoCoast Open](https://www.bgs.ac.uk/datasets/geocoast-open/).
- **Auth:** None for open layers.
- **Licence:** Open Government Licence; acknowledge BGS © UKRI.
- **Use:** Optional manual override via explicit `coastline_path=` for coastal inundation studies; default pipeline uses Boundary-Line MHW.

#### UKCEH Land Cover Map — **hardest (keep EIDC / manual for now)**

- **What exists:** EIDC catalogue GeoTIFF downloads (e.g. [LCM2023 10 m GB](https://catalogue.ceh.ac.uk/documents/7727ce7d-531e-4d77-b756-5cc59ff016bd)); separate **WMS** for viewing (e.g. LCM2023 10 m WMS on EIDC).
- **Auth / licence:** Raster LCM free for **non-commercial** use via EIDC ([UKCEH LCM page](https://www.ceh.ac.uk/data/ukceh-land-cover-maps); [raster licence](https://eidc.ceh.ac.uk/licences/lcm-raster)). **Commercial** use needs a paid / bespoke licence. Vector land parcels are separately licensed (admin fee for non-commercial).
- **Why not WMS for modelling:** WMS returns painted map images. MaxEnt needs the **classified GeoTIFF** (or equivalent). There is **no clean “fetch AOI as classified raster” public API** — keep EIDC download (or order) unless a licensed path is agreed.
- **Wiring cost:** High relative to OS/ONS; licence is the blocker for any productised / commercial reuse.

### Suggested wiring order

1. ~~**OS VectorMap** via Downloads API (no key)~~ — **wired** (`sdm/data/os_download.py`; Skipton → tiles SD+SE). Smoke artifact: `docs/artifacts/skipton-os-vmd-smoke.json`.
2. ~~**ONS boundaries** via Open Geography Portal FeatureServer / download~~ — **done:** `sdm/data/ons_download.py`, cache `data/raw/ons-boundaries/`.
3. ~~**Boundary-Line coastline** via Downloads API~~ — **wired** (`sdm/data/coastline_download.py`; Whitby smoke artifact).
4. **GeoCoast** retained as optional manual override for coastal-specific studies only.
5. Leave **CEH land cover** on EIDC / manual until the licence story is clear.

Do **not** put these vendor details in the stranger-facing README — this audit is the right home.

---

## Recommended next steps (for Building discussion)

1. ~~**Register granular Typer commands** or align docs~~ — **done:** docs match registered CLI (`sdm data` orchestrates all EV steps).
2. ~~**Fix merge inputs** in `sdm data`~~ — **done:** `build_ev_dataset_inputs` in `merge_ev_layers.py`.
3. ~~**Align OS raw path**~~ — **done:** raw `os-vector-map`, parquet cache `data/processed/os-data`.
4. **Enable `run_stats=True`** in climate step (or drop `climate_stats_*` from `variables_config.yml` until generated).
5. **Prioritise live API work** — ONS portal next (see Public API notes above); keep EA WCS (terrain, VOM) + WorldClim + OS VMD; plan tilearray for shared WCS tiling; defer CEH automation pending licence.
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
| Coastal | `commands/.../generate_coastal_distance.py` | `data/coastline_download.py` |
| OS | `commands/.../process_os_data.py` | `data/os.py`, `data/os_download.py` |
| Merge | `commands/.../merge_ev_layers.py` | — |
| Background | `commands/.../generate_background_points.py` | `occurrence/sampling.py` |

CLI orchestration: `sdm/cli.py` (`setup`, `data`, `background`, `pipeline`).
