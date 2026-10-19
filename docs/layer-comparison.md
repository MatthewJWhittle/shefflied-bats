# Environmental layer comparison (`sdm compare-layers`)

Tooling for [issue #38](https://github.com/MatthewJWhittle/shefflied-bats/issues/38): compare a **rebuilt** layer folder to the **current** layer set and write one table row per layer. Rebuild orchestration and live-path guards live in [#40](https://github.com/MatthewJWhittle/shefflied-bats/issues/40) / PR #44; this command only compares folders you already have.

## Implementation notes

Pass marks (per layer) follow the consolidated architecture review on #38:

1. **Same shape** — layer name, projection, cell size, grid origin, extent, no-data value and no-data cells must match exactly.
2. **Mostly the same values** — at least 95% of jointly valid cells are close. Exact match only for in-cell category and percentage layers stored as whole numbers. Continuous layers — including neighbourhood CEH landcover averages (`ceh_landcover_*_500m` / `_1000m` / `_1500m`), distances, heights, and climate — must be within 1% of the current layer’s full value range (architecture review decision on #43).
3. **Differences are local** — out-of-tolerance cells should form compact patches. Each compared layer gets a difference GeoTIFF plus a per-patch table (component id, cell count, bounding box, bbox fill ratio). Sparse / road-network-shaped components fail the local-patch check.
4. **Source changes are flagged, not failed** — optional `--source-versions` YAML. When before/after versions differ, the verdict is `flagged_source_change` even if values would otherwise pass.

Value-kind rules live in `is_exact_match_layer` / `is_ceh_neighbourhood_average` in `sdm/raster/layer_compare.py`.

Model checks (held-out AUC, cut-off map switch, brown long-eared roost marks from #29) are **not** run here. Pass `--model-hooks-csv` to write stub rows documenting those later checks after a real rebuild + retrain.

Layer order: union of variables across release-model configs under `data/sdm_config/*/variables_config.yml` (or repeatable `--priority-config`), then any other layers discovered in either folder.

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

# Or a custom file glob:
sdm compare-layers path/to/rebuilt path/to/current \
  -o layer_comparison/layer_comparison.csv \
  --glob 'evs-to-model.tif'

# Priority order from specific release models (repeatable):
sdm compare-layers path/to/rebuilt path/to/current \
  -o layer_comparison/layer_comparison.csv \
  --priority-config data/sdm_config/Plecotus_auritus_Roost/variables_config.yml \
  --priority-config data/sdm_config/Pipistrellus_pipistrellus_Roost/variables_config.yml
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
| `grid_check` | `ok` or which shape fields mismatched |
| `share_within_tolerance` | Fraction of jointly valid cells within tolerance |
| `max_difference` | Max absolute difference on jointly valid cells |
| `difference_map_path` | GeoTIFF of absolute differences |
| `differences_local` | Compact-patch heuristic (component count + bbox fill) |
| `patch_table_path` | Per-component CSV (id, cell count, bbox, fill ratio) |
| `source_version_*` / `reason` | Fillable / optional metadata |
| `verdict` | `pass`, `fail`, `flagged_source_change`, or `missing` |

## Tests

CI uses **small synthetic rasters only** (`tests/test_compare_ev_layers.py`, `tests/test_ev_output_guards.py`). No bucket access, credentials, or full downloads.
