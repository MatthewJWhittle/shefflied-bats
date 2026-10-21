# Overlapping environmental variables — thinned feature set (#72)

## Decision

Feature selection is thinned so models use fewer collinear / effort-proxy bands. Changes apply to:

- `data/ev_clusters_selected.csv` (`include` flags for the next `evs-to-model.tif` build)
- every committed `variables_config.yml` (root fallback, `data/sdm_config/*`, `data/sdm_tuning/*`)
- activity-type defaults in `sdm/models/maxent/maxent_model.py`
- multiscale defaults: `config.yml` `ev_stack.multiscale.windows_cells` and `DEFAULT_MULTISCALE_WINDOWS_CELLS` → **`[10]` only (500 m)**

Generation of source layers (climate, OS distances, VOM, land cover) is unchanged. The live `data/evs/` stack is not rewritten by this change; a later local rebuild into a non-production folder picks up the new selection.

## What was dropped

| Group | Dropped | Kept |
|-------|--------|------|
| Climate | `temp_ann_var`, `temp_mat_avg`, `prec_ann_var`, bio 2/3/7/8/9 | `temp_ann_avg`, bio_4 (seasonality), `prec_ann_avg`, `wind_ann_avg` |
| Survey-effort proxies | `os_distance_distance_to_minor_roads` | — |
| Buildings | `os_cover_buildings` | `os_distance_distance_to_buildings` (one building variable) |
| VOM | `vom_vegetation_height_mean` (+ scale copies) | `max` and `std` (cell + 500 m after rebuild) |
| Multiscale copies | 1000 m and 1500 m neighbourhood bands | cell + 500 m only |

Distance to coast is **not** changed here (see #75 / separate PR).

## Why

From the environmental-variables data-quality work and the ecology review on [#70](https://github.com/MatthewJWhittle/shefflied-bats/pull/70) (parent [#71](https://github.com/MatthewJWhittle/shefflied-bats/issues/71); this change is [#72](https://github.com/MatthewJWhittle/shefflied-bats/issues/72)):

- About 48 of 77 model-stack bands were neighbourhood scale copies; pairwise correlations between 500/1000/1500 m copies are typically |r| ≫ 0.7.
- Climate bands over this study area are highly collinear (temperature ≈ seasonality ≈ precipitation gradients).
- Distance to minor roads tracks access / recording effort (and correlates strongly with distance to buildings). Building cover and distance are both effort risks in roost models — keep one.

## Role

**Modeller** — feature selection and stack-build defaults only; ecology claims follow the review cited above.
