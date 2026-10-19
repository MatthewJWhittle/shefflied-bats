# Environmental layer comparison (`sdm compare-layers`)

Tooling for [issue #38](https://github.com/MatthewJWhittle/shefflied-bats/issues/38): compare a **rebuilt** layer folder to the **current** layer set and write one table row per layer. Rebuild orchestration and live-path guards live in [#40](https://github.com/MatthewJWhittle/shefflied-bats/issues/40); this command only compares folders you already have.

## Implementation notes

Pass marks (per layer) follow the consolidated architecture review on #38:

1. **Same shape** — layer name, projection, cell size, grid origin, extent, no-data value and no-data cells must match exactly.
2. **Mostly the same values** — at least 95% of jointly valid cells are close. Category and percentage-cover layers must match exactly; continuous layers (distances, heights, climate) must be within 1% of the current layer’s full value range.
3. **Differences are local** — out-of-tolerance cells should form a few compact patches. A difference GeoTIFF is written per compared layer for review; the `reason` column is left empty for a human note.
4. **Source changes are flagged, not failed** — optional `--source-versions` YAML records before/after source versions. When they differ, value mismatches become `flagged_source_change` instead of `fail`.

Model checks (held-out AUC, cut-off map switch, brown long-eared roost marks from #29) are **not** run here. Pass `--model-hooks-csv` to write stub rows documenting those later checks after a real rebuild + retrain.

Layer order: names listed in `data/sdm_config/Plecotus_auritus_Roost/variables_config.yml` first, then any other layers discovered in either folder (remaining candidates such as those discussed with #29 appear here once present as files).

## How to run

```bash
# After a rebuild has written to a separate folder (never overwriting the current set):
sdm compare-layers path/to/rebuilt/evs path/to/current/evs \
  -o data/evs/layer_comparison.csv \
  --difference-maps-dir data/evs/layer_diff_maps \
  --model-hooks-csv data/evs/model_comparison_stub.csv
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
| `differences_local` | Compact-patch heuristic |
| `source_version_*` / `reason` | Fillable / optional metadata |
| `verdict` | `pass`, `fail`, `flagged_source_change`, or `missing` |

## Tests

CI uses **small synthetic rasters only** (`tests/test_compare_ev_layers.py`). No bucket access, credentials, or full downloads.
