# Environmental layer comparison (`sdm compare-layers`)

Tooling for [issue #38](https://github.com/MatthewJWhittle/shefflied-bats/issues/38): compare a **rebuilt** layer folder to the **current** layer set and write one table row per layer. Rebuild orchestration and live-path guards live in [#40](https://github.com/MatthewJWhittle/shefflied-bats/issues/40) / PR #44; this command only compares folders you already have.

## Implementation notes

The objective is **closeness, not exactness** (revised architecture review on #38). Per layer:

1. **Grid (precondition)** — rebuilt must sit on the same grid as current (projection, cell size, origin / alignment, extent) so cells can be compared. Use `--align` to crop / snap the rebuilt stack onto the current reference grid (`rioxarray.reproject_match`) when shapes differ. No-data **cell** mismatches are **not** a grid failure.
2. **Continuous layers** — cell-wise `(rebuilt − current) / layer_range` (range from the current layer’s jointly valid min/max). Report mean, median, 95th percentile and max of the absolute normalised difference, the share of cells within **1% of range**, and the **correlation**. Pass when at least **95%** of comparable cells are within 1% of range.
3. **Category / percentage layers** — report the share of cells with the same value. Pass when at least **95%** match.
4. **No-data** — cells empty in one layer but not the other count as **not-close** in the share figure, and the mismatch count is reported in its own column (`n_nodata_mismatch`). Cells empty in both are excluded.
5. **Differences local** — not-close cells should form compact patches (component count + bbox fill). Each layer gets a difference GeoTIFF and a per-patch table.
6. **Intended method changes** — configurable `--flag-layer` / `--flag-prefix` (e.g. `os_distance` after the #42 edge-distance change). Matching layers are reported as **`flagged`** with stats, **not** as failures.
7. **Source changes** — optional `--source-versions` YAML. When before/after versions differ, verdict is `flagged_source_change`.

Value-kind rules live in `is_exact_match_layer` / `is_neighbourhood_average` in `sdm/raster/layer_compare.py`.

**Engine:** both sides are opened lazily with **rioxarray + dask chunks**, compared with vectorised xarray ops (`(new − old) / range`, `.mean` / `.median` / `.quantile`, `xr.corr`, combined `notnull` masks), and reduced with a **single** `dask.compute()`. Patches use `scipy.ndimage.label` on the computed boolean array. No per-cell / per-tile Python loops.

Model checks (held-out AUC, cut-off map switch, brown long-eared roost marks from #29) are **not** run here. Pass `--model-hooks-csv` to write stub rows documenting those later checks after a real rebuild + retrain.

Layer order: union of variables across the 18 release-model `variables_config.yml` files under `data/sdm_config/*/variables_config.yml` (or repeatable `--priority-config`), then any other layers discovered in either folder.

Without `--stack-only` / `--glob`, duplicate layer names across files are resolved by preferring `evs-to-model.tif` (with a warning). Ambiguous duplicates raise an error.

Comparison outputs reuse the same write guard as the rebuild command (`sdm.utils.ev_output_guards` from #44): refuse writes under `data/evs`, any `inputs/` path, `${INPUT_DATA_ROOT}`, and bucket inputs URIs.

## How to run

```bash
# After a rebuild has written to a separate folder (never overwriting the current set):
sdm compare-layers path/to/rebuilt/evs path/to/current/evs \
  -o layer_comparison/layer_comparison.csv \
  --difference-maps-dir layer_comparison/difference_maps \
  --model-hooks-csv layer_comparison/model_comparison_stub.csv

# Prefer the merged stack only (ignore intermediate single-band files):
sdm compare-layers path/to/rebuilt path/to/current \
  -o layer_comparison/layer_comparison.csv \
  --stack-only

# Crop / snap rebuilt onto the current grid when extents differ:
sdm compare-layers path/to/rebuilt path/to/current \
  -o layer_comparison/layer_comparison.csv \
  --align

# Flag intended method-change layers (e.g. OS distance after #42):
sdm compare-layers path/to/rebuilt path/to/current \
  -o layer_comparison/layer_comparison.csv \
  --flag-prefix os_distance
```

Optional source-version file (`source_versions.yml`):

```yaml
os_distance_distance_to_minor_roads:
  current: "OS OpenMap 2023"
  rebuilt: "OS OpenMap 2024"
```

Folders may contain single-band GeoTIFFs (stem or band description = layer name) and/or multi-band stacks (band descriptions = layer names). Names are tidied the same way as the merge pipeline.

## Output columns

| Column | Meaning |
|--------|---------|
| `layer_name` | Tidied layer / band name |
| `value_kind` | `exact` or `continuous` |
| `grid_check` | `ok` / `ok (aligned)` or which shape fields mismatched |
| `share_within_tolerance` | Share of comparable cells that are close (nodata mismatches count as not-close) |
| `n_nodata_mismatch` | Cells empty in one side only |
| `norm_abs_diff_*` | Mean / median / p95 / max of `\|diff\| / range` (continuous only) |
| `correlation` | `xr.corr` over spatial dims |
| `max_difference` | Max absolute difference on jointly valid cells |
| `difference_map_path` | GeoTIFF of absolute differences |
| `differences_local` | Compact-patch heuristic (component count + bbox fill) |
| `patch_table_path` | Per-component CSV (id, cell count, bbox, fill ratio) |
| `source_version_*` / `reason` | Fillable / optional metadata |
| `verdict` | `pass`, `fail`, `flagged`, `flagged_source_change`, or `missing` |

## Tests

CI uses **small synthetic rasters only** (`tests/test_compare_ev_layers.py`). No bucket access, credentials, or full downloads.
