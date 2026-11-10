# Oct 25 map features × current-main training

## Implementation notes

Release configs that restore the **environmental feature lists** used for the **25 October 2026** published suitability maps, while keeping **today’s training methods** on `main` (CV, cut-offs, MaxEnt adapter / `sdm train`, background knobs). Separate from architecture grill / Zarr / tile-store / rebuild redesign.

### Canonical 25 Oct maps

| Piece | Value |
| --- | --- |
| Published prediction prefix | `gs://habitat-suitability-modelling-data/predictions/<model_id>/20261025_28297ec_cfg*/` |
| Git tip encoded in version id | `28297ec` (merge of #104, versioned prediction publish) |
| Config source for feature lists | `data/sdm_config/*/variables_config.yml` at `28297ec` (unchanged by #104; same lists as Oct 24 MLflow tip `f833550`) |
| Selection evidence | Project QGIS reference `hsm-aerial-latest.qgs`: abstract *“Latest prediction selected 25 October 2026”* for each `20261025_28297ec_*` layer; provenance `…__run.json` |

**Brown long-eared roost.** Keep the Oct 25 list (12 layers including `vom_vegetation_height_mean` + `vom_vegetation_height_std`). Do **not** use later woody/ALAN nine-layer, BLEM*, or FINAL3 swaps. Later “latest” pin `20261026_a7e1087_*` (PR #128 compare/main tables) is **not** the 25 Oct selection.

### Hypothesis (verify / discard)

| Claim | Verdict |
| --- | --- |
| 25 Oct maps are baseline-`main` training published under `predictions/` ~2026-10-25 with tip `28297ec` | **Verified** — version folders and QGIS abstracts match |
| Same-day #103 10k/50k comparison-points trial `20261025_2f2d318_cfg*` | **Discarded** for these maps — published folders use `28297ec`, not `2f2d318` (that SHA is not a commit on this repo); #103 trial packaging is `model_config.background-as-drawn.yml` (#110), not these maps |
| Training used a 23 Oct EV stack pre-#83/#84 | **Unresolved for live rasters** (no ADC here to CRC the Oct 25 EV). **Code tip:** #83 and #84 merged morning 24 Oct and are ancestors of the Oct 24 retrain tip `f833550` / map tip `28297ec` |

### Training methods: Now / Instead

| Piece | Now (this branch = current `main`) | Instead (original 25 Oct published runs) |
| --- | --- | --- |
| Feature lists (`variables_config.yml`) | Restored to Oct 25 (`28297ec`) | Same |
| CV | Blocked **10 km** × **5** folds, seed **42** | Same (already default by 25 Oct) |
| Final fit | MaxEnt on **all** training rows after CV | Same |
| Cut-off | Pooled OOF percentile **p10**; no cut-off if `model_limited` (whiskered in-flight) | Same METHOD |
| AUC gate on cut-off | Current `main` threshold code (METHOD.md) | Same era |
| Background | Live-train knobs under `model.background` (8000, contrast, σ 6.5, 500 m exclusion, 2 km thin, 10× cap, distance weights) — defaults = today’s behaviour | Same behaviour; YAML packaging of knobs landed later (#110) |
| Trainer | Current MaxEnt adapter / `sdm train` | Pre-adapter MaxEnt path (same elapid defaults) |
| Woody / canopy / height-band features (#121/#122/#124) | **Not** in species lists (restored off) | Not present |
| BLE suburban_500m restore / woody swap | **Not** applied | Not present on 25 Oct list |

### Feature match (all 17 trainable models)

| Model | Features on this branch | Features on 25 Oct maps | Match |
| --- | --- | --- | --- |
| `Myotis_brandtii_Roost` | `os_distance_distance_to_buildings`, `climate_bioclim_bio_9`, `climate_stats_wind_ann_avg`, `vom_vegetation_height_std_500m`, `ceh_landcover_urban_1500m`, `ceh_landcover_arable_1500m`, `ceh_landcover_suburban`, `ceh_landcover_improved_grassland`, `terrain_stats_twi`, `terrain_dtm`, `os_cover_water_500m` | same (`28297ec`) | yes |
| `Myotis_daubentonii_In_flight` | `climate_bioclim_bio_2`, `climate_stats_temp_mat_avg`, `climate_bioclim_bio_9`, `climate_stats_prec_ann_var`, `terrain_stats_roughness`, `climate_bioclim_bio_8`, `ceh_landcover_arable`, `vom_vegetation_height_max`, `ceh_landcover_suburban`, `os_cover_water`, `os_cover_water_500m`, `ceh_landcover_urban`, `ceh_landcover_wetland_1500m`, `ceh_landcover_broadleaved_woodland_500m`, `ceh_landcover_improved_grassland_1000m` | same | yes |
| `Myotis_daubentonii_Roost` | `vom_vegetation_height_max_1000m`, `ceh_landcover_upland_heathland_500m`, `vom_vegetation_height_mean_1500m`, `ceh_landcover_broadleaved_woodland_500m`, `ceh_landcover_suburban_500m`, `ceh_landcover_wetland_500m`, `terrain_dtm`, `climate_bioclim_bio_9`, `climate_bioclim_bio_8`, `vom_vegetation_height_std`, `os_cover_buildings`, `ceh_landcover_arable_500m`, `os_cover_water_500m` | same | yes |
| `Myotis_mystacinus_In_flight` | `climate_stats_temp_ann_var`, `terrain_dtm`, `climate_stats_temp_mat_avg`, `os_distance_distance_to_buildings`, `climate_bioclim_bio_8`, `terrain_stats_slope`, `vom_vegetation_height_std`, `terrain_stats_tpi`, `os_cover_water`, `terrain_stats_twi`, `ceh_landcover_coniferous_woodland`, `os_distance_distance_to_water`, `terrain_stats_aspect_northness_slope`, `ceh_landcover_arable_500m`, `vom_vegetation_height_max_500m`, `ceh_landcover_broadleaved_woodland_500m`, `os_cover_water_500m`, `ceh_landcover_urban_500m` | same | yes |
| `Myotis_mystacinus_Roost` | `climate_stats_prec_ann_var`, `climate_stats_temp_ann_avg`, `os_distance_distance_to_buildings`, `climate_bioclim_bio_9`, `terrain_stats_slope`, `climate_stats_wind_ann_avg`, `vom_vegetation_height_max`, `ceh_landcover_upland_heathland_500m`, `ceh_landcover_arable_500m`, `ceh_landcover_broadleaved_woodland_500m` | same | yes |
| `Myotis_nattereri_In_flight` | `terrain_dtm`, `climate_bioclim_bio_9`, `ceh_landcover_arable`, `os_distance_distance_to_buildings`, `climate_stats_wind_ann_avg`, `vom_vegetation_height_max`, `vom_vegetation_height_mean_1000m`, `terrain_stats_aspect_eastness_slope`, `ceh_landcover_arable_500m`, `ceh_landcover_broadleaved_woodland_1000m`, `ceh_landcover_urban_500m`, `ceh_landcover_suburban_1000m` | same | yes |
| `Myotis_nattereri_Roost` | `os_distance_distance_to_buildings`, `ceh_landcover_arable`, `climate_bioclim_bio_3`, `vom_vegetation_height_max`, `ceh_landcover_suburban_1000m`, `vom_vegetation_height_max_1000m`, `ceh_landcover_improved_grassland_500m` | same | yes |
| `Nyctalus_leisleri_In_flight` | `climate_stats_temp_ann_avg`, `climate_bioclim_bio_9`, `terrain_stats_slope`, `os_cover_water`, `ceh_landcover_improved_grassland`, `ceh_landcover_broadleaved_woodland_500m`, `ceh_landcover_suburban_500m`, `ceh_landcover_improved_grassland_500m`, `ceh_landcover_upland_heathland_1500m` | same | yes |
| `Nyctalus_leisleri_Roost` | `climate_bioclim_bio_4`, `climate_bioclim_bio_9`, `climate_stats_wind_ann_avg`, `vom_vegetation_height_std`, `terrain_stats_aspect_eastness_slope`, `ceh_landcover_broadleaved_woodland_500m`, `ceh_landcover_suburban`, `ceh_landcover_improved_grassland_1500m`, `ceh_landcover_coniferous_woodland_500m` | same | yes |
| `Nyctalus_noctula_In_flight` | `climate_stats_prec_ann_var`, `climate_stats_temp_mat_avg`, `vom_vegetation_height_std`, `os_distance_distance_to_buildings`, `ceh_landcover_suburban`, `os_distance_distance_to_water`, `ceh_landcover_wetland_1500m`, `ceh_landcover_suburban_1000m` | same | yes |
| `Nyctalus_noctula_Roost` | `terrain_dtm`, `os_distance_distance_to_minor_roads`, `vom_vegetation_height_max`, `ceh_landcover_coniferous_woodland`, `os_distance_distance_to_buildings`, `os_cover_water`, `ceh_landcover_arable`, `vom_vegetation_height_std_500m` | same | yes |
| `Pipistrellus_pipistrellus_In_flight` | `climate_bioclim_bio_7`, `climate_stats_temp_mat_avg`, `os_distance_distance_to_buildings`, `climate_bioclim_bio_9`, `vom_vegetation_height_max`, `vom_vegetation_height_mean_1000m`, `vom_vegetation_height_max_1000m`, `ceh_landcover_suburban`, `ceh_landcover_suburban_1500m`, `ceh_landcover_broadleaved_woodland_1500m`, `ceh_landcover_arable`, `os_cover_water`, `os_distance_distance_to_water` | same | yes |
| `Pipistrellus_pipistrellus_Roost` | `climate_bioclim_bio_9`, `os_distance_distance_to_buildings`, `vom_vegetation_height_std`, `vom_vegetation_height_mean`, `ceh_landcover_suburban`, `os_distance_distance_to_water`, `terrain_stats_twi`, `ceh_landcover_coniferous_woodland`, `ceh_landcover_suburban_500m`, `ceh_landcover_arable_500m`, `ceh_landcover_suburban_1500m` | same | yes |
| `Pipistrellus_pygmaeus_In_flight` | `climate_bioclim_bio_7`, `climate_stats_prec_ann_var`, `os_distance_distance_to_minor_roads`, `os_distance_distance_to_buildings`, `vom_vegetation_height_mean`, `vom_vegetation_height_std`, `vom_vegetation_height_std_1500m`, `terrain_stats_roughness`, `ceh_landcover_urban`, `ceh_landcover_suburban`, `ceh_landcover_grassland_500m`, `ceh_landcover_arable`, `ceh_landcover_arable_1500m`, `os_cover_water`, `ceh_landcover_improved_grassland` | same | yes |
| `Pipistrellus_pygmaeus_Roost` | `climate_bioclim_bio_9`, `os_distance_distance_to_buildings`, `vom_vegetation_height_mean_500m`, `vom_vegetation_height_std`, `terrain_stats_twi`, `ceh_landcover_grassland_500m`, `ceh_landcover_suburban`, `ceh_landcover_suburban_1500m`, `ceh_landcover_arable_1500m` | same | yes |
| `Plecotus_auritus_In_flight` | `climate_stats_temp_ann_avg`, `os_distance_distance_to_buildings`, `climate_bioclim_bio_9`, `vom_vegetation_height_max`, `vom_vegetation_height_max_1000m`, `vom_vegetation_height_mean_1000m`, `ceh_landcover_suburban`, `os_cover_buildings`, `ceh_landcover_grassland_500m`, `ceh_landcover_arable_500m`, `ceh_landcover_broadleaved_woodland_1000m`, `ceh_landcover_suburban_1500m`, `os_cover_water_500m` | same | yes |
| `Plecotus_auritus_Roost` | `os_distance_distance_to_minor_roads`, `terrain_stats_slope`, `os_distance_distance_to_major_roads`, `ceh_landcover_arable_1000m`, `vom_vegetation_height_mean`, `os_cover_buildings`, `ceh_landcover_grassland_1000m`, `climate_bioclim_bio_2`, `climate_stats_temp_ann_avg`, `ceh_landcover_broadleaved_woodland_500m`, `ceh_landcover_urban_1500m`, `vom_vegetation_height_std` | same | yes |

`Pipistrellus_nathusii_Roost` config still exists but does not train (below min presence); left unchanged (already matched Oct 25).

### Train + package for hsm-app

Use a **new versioned output path** (do not overwrite live production layers/predictions). Requires national EV stack + occurrence GeoJSON (not present in cloud-agent checkout). Contract: [model-package-contract.md](model-package-contract.md).

```bash
# 0) Inputs (local or GCS via INPUT_DATA_ROOT) — EV must include Oct 25 bands
#    (legacy vom_vegetation_height_* still include:1 in ev_clusters_selected.csv).
#    Prefer the EV stack that matches the maps you intend to ship; do not use
#    inputs/data/evs/versions/2026-11-05-woody-alan-edge/ unless that stack
#    still carries every listed Oct 25 band.

# 1) Train all 17 (blocked 10 km × 5, seed 42, pooled OOF p10)
export MODEL_OUTPUT_ROOT="data/sdm_models/runs"   # or a dated scratch / GCS runs prefix
uv run sdm train \
  --bats-file data/processed/bats-tidy.geojson \
  --ev-file data/evs/evs-to-model.tif

# Note the new run id under data/sdm_models/runs/<UTC>-<hex>/
# Each model: {model_id}/model.pkl + package.json + thresholds.json
# Run: model_results.csv, thresholds.csv, thresholds.json

# 2) Predict (native EPSG:27700 COGs)
uv run sdm predict \
  --run <run-id> \
  --ev-path data/evs/evs-to-model.tif
# optional local publish only if Matt okays bucket write:
#   uv run sdm predict --run <run-id> --ev-path data/evs/evs-to-model.tif --publish

# 3) Web Mercator COGs for hsm-app
mkdir -p exports/oct25-features-current-training
uv run sdm export-rasters data/sdm_predictions/prediction_*.tif \
  -o exports/oct25-features-current-training \
  --output-crs EPSG:3857 --cog

# EV driver COG (once per project) — same export pattern on evs-to-model.tif
uv run sdm export-rasters data/evs/evs-to-model.tif \
  -o exports/oct25-features-current-training \
  --output-crs EPSG:3857 --cog

# 4) ModelMetadata JSON per package (hsm-app multipart `metadata`)
uv run python scripts/build_model_metadata_from_package.py \
  data/sdm_models/runs/<run-id>/<model_id>/package.json \
  -o exports/oct25-features-current-training/<model_id>_metadata.json

# 5) Upload artefacts to MatthewJWhittle/hsm-app (HTTP) —
#    EV COG → project; per model: suitability COG + metadata + model.pkl
#    See docs/hsm-visualiser-integration.md. Do not push to gs:// unless Matt okays.
```

### Out of scope

- Tile-store, Zarr, rebuild redesign (architecture grill C4–C8)
- Uploading to `gs://` or overwriting live production predictions
- Invented accuracy numbers or maps (training not run in this environment)
