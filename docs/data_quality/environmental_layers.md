# Rebuild-5 environmental layer data quality

Study-area aggregate checks on the **rebuild-5** environmental stack (the current layer set after the EV rebuild). Use this when refining models and feature data quality. It is separate from the ecology reference.

**Scope.** Figures below are study-area aggregates only: no record points, no presence-vs-background tables, no site names. Licensed rasters and training parquet stay out of git.

**Sources.** Aggregates were computed from the rebuild-5 `evs-to-model.tif` (77 float32 bands, EPSG:27700, 100 m cells, nodata = NaN). Committed tables:

| File | Contents |
|------|----------|
| [`study_area_stats.csv`](study_area_stats.csv) | Per-band valid count, missing share, min / percentiles / max, mean, sd |
| [`landcover_shares.csv`](landcover_shares.csv) | CEH land-cover mean/median cover and non-zero cell shares by scale |

Regenerate (when you have a local stack path; do not commit the raster):

```bash
uv run python scripts/summarise_ev_study_area_stats.py \
  --stack /path/to/evs-to-model.tif \
  --output-dir docs/data_quality
```

---

## Per-layer study-area summary

Full range and missing shares are in [`study_area_stats.csv`](study_area_stats.csv). Selected bands:

| Variable | Missing % | Min | Median | Max | Mean | Notes |
|----------|----------:|----:|-------:|----:|-----:|-------|
| `terrain_dtm` | 6.81 | −9.01 | 88.5 | 730.3 | 141.1 | Min unexpected inland (see issues) |
| `terrain_stats_slope` | 6.99 | 0 | 0.035 | 0.737 | 0.057 | Radians; plausible vs old set |
| `terrain_stats_twi` | 6.99 | 4.97 | 9.05 | 31.0 | 9.73 | Usual TWI range; old set was off scale |
| `terrain_stats_roughness` | 6.75 | 0 | 0.0105 | 0.269 | 0.0169 | Smaller than old set |
| `terrain_stats_tpi` | 6.81 | −41.9 | 0.0015 | 29.9 | ~0 | Smaller range than old set |
| `vom_vegetation_height_max` | 9.97 | 0 | 3.75 | 61.5 | 5.94 | Max above likely tree height |
| `os_cover_water` | 0.001 | 0 | 0 | 100 | 9.66 | Mean ~9.7%; p95 = 100 |
| `os_distance_distance_to_water` | 0 | 0 | 113 | 4376 | 221 | Denser water than old set |
| `bgs_coast_distance_to_coast` | 0.008 | 0 | 7000 | 7000 | 6228 | Cap at 7000 m; most cells at cap |
| `os_distance_distance_to_buildings` | 0 | 0 | 284 | 7000 | 594 | Cap at 7000 m |
| `os_distance_distance_to_major_roads` | 0 | 1.5 | 3270 | 7000 | 3673 | Cap at 7000 m |
| Climate bands (e.g. `climate_stats_temp_ann_avg`) | 7.97 | — | — | — | — | Missing ~8%; values essentially unchanged vs old set |

Land-cover bands are cover fractions on a 0–10 000 scale (percent × 100). Cell-scale class means sum to about 91% of the area; see [`landcover_shares.csv`](landcover_shares.csv).

---

## Quality issues to investigate

Each item has evidence from the study-area aggregates and a suggested check. None of these are presence/background contrasts.

### 1. Elevation minimum about −9 m

**Evidence.** `terrain_dtm` min = −9.01 m (`study_area_stats.csv`). The study area is inland South Yorkshire plus buffer; a negative cell mean is unexpected for DTM over that extent. Max ~730 m is plausible for western upland cells (e.g. Dales peaks near the buffer).

**Suggested check.** Map cells with elevation &lt; 0 m against published LiDAR / OS terrain; confirm whether they are coastal fringe, quarries, or edge/nodata artefacts after merge.

### 2. Vegetation height reaching about 61 m

**Evidence.** `vom_vegetation_height_max` max = 61.5 m; p95 = 19.5 m. Heights above ~40–45 m are unlikely for trees in this region, so a few cells are probably structures or VOM artefacts.

**Suggested check.** Locate cells above a threshold (e.g. 40 m) on the VOM source; decide whether to clip, mask, or leave as-is for MaxEnt.

### 3. Much denser water layer than the old set

**Evidence.** `os_distance_distance_to_water` median 113 m (was ~209 m on the previous stack), p95 750 m (was ~2 600 m), max 4 376 m (was ~13 km). Rebuild-5 likely includes smaller watercourses.

**Suggested check.** Compare water-feature selection / dissolve rules in the OS step against the previous stack; confirm intentional inclusion of drains and minor channels.

### 4. Slope and wetness now on plausible scales (fixed)

**Evidence.** Rebuild-5 `terrain_stats_slope` median 0.035 rad (~2°), max 0.74 rad (~42°). `terrain_stats_twi` runs ~5–31 with median ~9. The previous stack had slope medians near 1.3 rad (~73°) and TWI roughly −0.4–14 with median ~1 — not usable as terrain.

**Suggested check.** Treat rebuild-5 slope/TWI as the trusted scales for ecological interpretation; do not reuse old-set slope/TWI separations.

### 5. Water-cover layer looks inflated

**Evidence.** `os_cover_water` mean ≈ 9.7%, p95 = 100%, max = 100%. A study-area mean near 10% open water is high for South Yorkshire plus buffer if the layer is meant to be open-water fraction.

**Suggested check.** Compare against a published water mask (e.g. OS OpenMap / CEH water bodies / Ordnance Survey water area statistics) on the same grid; inspect whether the cover metric counts linear water as full-cell cover.

### 6. Distance-to-coast as a likely proxy for Nathusius' pipistrelle

**Evidence.** `bgs_coast_distance_to_coast` is in the Nathusius' pipistrelle roost feature list (`data/sdm_config/Pipistrellus_nathusii_Roost/variables_config.yml`). On this inland study area, median and p5–p95 are all at the 7 000 m cap (mean ~6 228 m), so the layer has little within-area gradient except near the eastern/northern fringe.

**Suggested check.** Feature-selection review: coastal distance has limited ecological meaning this far inland and may act as a spatial proxy; consider dropping or replacing it for inland models.

### 7. Nathusius' pipistrelle gap in attached per-species material

**Evidence.** The working notes that accompanied these aggregates summarised per-species presence-vs-background contrasts for other models; **Nathusius' pipistrelle was not included in that attached material**. No presence or background counts for that species are published here.

**Suggested check.** When regenerating any per-species diagnostics offline, confirm whether `Pipistrellus_nathusii_Roost` was trained on this stack and include it deliberately. Do not invent counts.

### 8. Distance layers capped at 7 000 m

**Evidence.** Coast, buildings, and major-road distances are clipped to 0–7 000 m in the model stack (`build_model_ev_stack` layer clipping; see also `docs/rebuild-ev-layers.md`). In the aggregates, coast and major roads sit at 7 000 m for a large share of cells; buildings max = 7 000 m.

**Effect.** Beyond 7 km the layers are flat, so models cannot distinguish “far” from “very far”. That is intentional clipping for the study buffer, but it compresses gradients for species that respond at longer ranges.

**Suggested check.** Keep the cap documented in model cards; revisit only if a species hypothesis needs longer-range distance.

### 9. Old-set slope / wetness scale bug is fixed in rebuild-5

**Evidence.** Same as issue 4: rebuild-5 slope and TWI match expected physical ranges; the previous stack did not.

**Suggested check.** Prefer rebuild-5 for any re-tuning that uses terrain; treat old-set terrain coefficients as non-comparable.

---

## Implementation notes

- Aggregates only; regenerator script does not read training parquet.
- Figures taken from the rebuild-5 study-area run that produced the committed CSVs (20 Oct 2026 method notes, sanitized).
- For ecology and species natural history, use the separate ecology reference — not this file.
