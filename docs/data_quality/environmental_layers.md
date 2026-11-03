# Environmental variables — layer data quality

Study-area aggregate checks on **the environmental variables** stack (the current layer set). Use this when refining models and feature data quality. It is separate from the ecology reference.

**Scope.** Study-area aggregates only: no record points, no presence-vs-background tables by location, no site or roost names, no grid references. Licensed rasters and training parquet stay out of git. Do not commit histogram or map image outputs from the QA script.

---

## Key finding

**The study-area boundary is buffered into the sea and Humber estuary.** About **6%** of cells sit on the coastline or in the sea (`bgs_coast_distance_to_coast` = 0). That single fact explains most of the odd headline numbers:

| Symptom | How the sea / estuary buffer explains it |
|---------|------------------------------------------|
| Inflated water cover (`os_cover_water` mean **9.7%**) | **77%** of cells at 100% water are sea/estuary inside the buffer. Excluding coast cells, mean water drops to **4.3%**; beyond 1 km of the coast, **2.9%** (closer to the ~1–3% inland expectation). |
| Negative elevations (`terrain_dtm` min **−9.0 m**; 43 915 cells &lt; 0) | **90%** of negative cells are within 2 km of the coast (foreshore, estuary, sea). ~4 500 cells remain negative further inland (low drained land and/or DTM artefacts). |
| Distance-to-coast saturates | **85%** of cells sit at the **7 000 m** cap; the layer only varies along the coastal strip. |

**Suggested fix (root cause).** Clip the study boundary to land (mean high water) before building layers, and/or mask sea cells from modelling and background sampling. That also shrinks the climate and vegetation missing rings that follow the sea buffer.

---

## Sources and regeneration

Aggregates from the environmental variables `evs-to-model.tif` (77 float32 bands, EPSG:27700, 100 m cells, nodata = NaN).

| File | Contents |
|------|----------|
| [`layer_stats.csv`](layer_stats.csv) | Per-band min / percentiles / max / mean, % missing, cells at max, verdict + reason |
| [`study_area_stats.csv`](study_area_stats.csv) | Earlier percentile table (includes p25/p75/sd) |
| [`landcover_shares.csv`](landcover_shares.csv) | CEH land-cover mean/median and non-zero shares by scale |

```bash
uv run python scripts/make_layer_qa.py \
  --stack /path/to/evs-to-model.tif \
  --output-dir /tmp/layer_qa \
  --print-coast-followup
# Optional local histograms (do not commit):
#   ... --write-histograms
```

Also available: `scripts/summarise_ev_study_area_stats.py` for the simpler study-area / land-cover CSVs.

---

## Per-layer summary

Full table: [`layer_stats.csv`](layer_stats.csv) (verdict `ok` / `check` / `flag`). Selected bands:

| Variable | Missing % | Min | Median | Max | Mean | Verdict |
|----------|----------:|----:|-------:|----:|-----:|---------|
| `terrain_dtm` | 6.81 | −9.01 | 88.5 | 730.3 | 141.1 | flag — negatives mostly coastal buffer |
| `terrain_stats_slope` | 6.99 | 0 | 0.035 | 0.737 | 0.057 | ok — radians (see below) |
| `terrain_stats_twi` | 6.99 | 4.97 | 9.05 | 31.0 | 9.73 | check — typical TWI range |
| `vom_vegetation_height_max` | 9.97 | 0 | 3.75 | 61.5 | 5.94 | flag — 37 cells &gt; 40 m |
| `os_cover_water` | 0.001 | 0 | 0 | 100 | 9.66 | flag — inflated by sea buffer |
| `os_cover_water_500m` / `_1000m` / `_1500m` | 1.9–6.1 | 0 | ~2 | 100 | 8.5–9.3 | flag — same pattern |
| `bgs_coast_distance_to_coast` | 0.008 | 0 | 7000 | 7000 | 6228 | flag — 85% at 7000 m cap |
| `os_distance_distance_to_major_roads` | 0 | 1.5 | 3270 | 7000 | 3673 | flag — 25% at cap |
| `os_distance_distance_to_buildings` | 0 | 0 | 284 | 7000 | 594 | flag — 0.24% at cap (low impact) |
| Climate bands | 7.97 | — | — | — | — | check — missing ring at coast / edge |
| Vegetation height (focal scales) | 10–18 | — | — | — | — | check — coverage gap + sea; rises with radius |

Land-cover bands use 0–10 000 units (percent × 100) and are in range; see [`landcover_shares.csv`](landcover_shares.csv).

---

## Flagged issues (cause and suggested fix)

| Layer | Key number | Likely cause | Suggested fix |
|-------|------------|--------------|---------------|
| Water cover (`os_cover_water` + 500/1000/1500 m) | Mean 9.7%; 7.3% of cells at 100% | Sea and Humber estuary buffer (~6% of cells at coast distance 0); 77% of 100% water cells are that buffer | Clip boundary to land (MHW) before building layers, or mask sea cells from model and background sampling |
| Elevation (`terrain_dtm`) | Min −9.0 m; 43 915 cells &lt; 0 | 90% within 2 km of the coast; ~4 500 further inland (drained land / quarry / DTM artefacts) | Mask sea as above; spot-check inland negatives; optionally floor-clip (e.g. −3 m) if artefacts |
| Vegetation max height | Max 61.5 m; **37 cells &gt; 40 m** | Sparse single cells: pylons, chimneys, masts, or tall buildings in the vegetation model | Cap at ~40 m or mask with a buildings/structures layer; 37 cells is low impact |
| Distance to coast | **85%** of cells at 7 000 m cap | Cap at 7 km while most of the area is further inland; layer only varies on the coastal strip | Drop from inland species models (including the Nathusius' pipistrelle config) or remove the cap |
| Distance to major roads | 25.4% at 7 000 m cap | Uplands far from major roads; cap flattens them | Raise/remove the cap or log-transform |
| Distance to buildings | 0.24% at 7 000 m cap | Same cap, low impact | Fine as is; note only |
| Climate (all) | 8.0% missing | Climate grid ends at the coast and boundary edge — a missing ring, mostly the sea buffer | Fixed by clipping boundary to land; otherwise fill edge cells from nearest |
| Vegetation height (all) | 10–18% missing (more at larger radii) | Vegetation coverage gap plus sea; a **north-east block also lacks elevation** (DTM gap) | Check source tiles for the north-east elevation gap |

Checked and OK (or expected): slope (see units), TWI 5.0–31.0, TPI, aspect, roughness, land-cover classes.

### Slope units (confirmed in code)

`terrain_stats_slope` range 0–0.74 reads as **radians** (max ≈ 42°), which is plausible at 100 m. The rebuild path calls `generate_terrain_stats` without overriding units; the default is radians:

- `sdm/commands/data_preparation/environmental/generate_terrain_stats.py` — `output_slope_units: str = "radians"`
- `sdm/raster/terrain.py` — `process_dem_to_terrain_attributes(..., output_slope_units: str = "radians")`; slope is computed in radians and only converted if degrees/percent are requested

Treat slope/TWI from the environmental variables as the trusted scales (the previous stack’s slope/TWI were off scale).

### Distance caps at 7 000 m

Coast, buildings, and major-road distances are clipped to 0–7 000 m in the model stack (`build_model_ev_stack` layer clipping; see `docs/rebuild-ev-layers.md`). Beyond 7 km the layers are flat.

---

## Nathusius' pipistrelle (training gap, not a layer bug)

Records exist in the tidy records file (47 after the accuracy filter: 32 roost, 15 in flight). They fail the **15-presence minimum after one-per-cell thinning**: the 32 roost records fall in **3** distinct cells and the 15 in-flight records in **11**. Both are below 15, so no model and no presence-vs-background stats for this species on the environmental variables run. That is a data-volume limit, not an environmental-layer defect.

The training log wrongly reports **0** presence rather than 3 / 11. Likely cause: small-sample sample-weighting producing NaNs that `dropna()` removes. Worth a small logging fix so the real thinned counts appear. The existing `Pipistrellus_nathusii_Roost` config is from an earlier run; distance-to-coast in that feature list should be reviewed for inland models (see table above).

---

## Implementation notes

- Aggregates only; QA script does not read training parquet and does not write presence tables.
- Histogram/map image outputs from the QA script must not be committed (licensed layers / local inspection only).
- For ecology and species natural history, use the separate ecology reference — not this file.
