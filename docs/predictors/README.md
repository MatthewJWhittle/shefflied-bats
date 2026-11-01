# Predictor layer documentation

Human-readable pages and machine-readable YAML for every environmental predictor
band used by the public `sdm data` pipeline on `main`. Docs travel with layers
via raster attrs and an optional sidecar. Upstream datasets have their own
source docs (URL, licence, citation).

## Why this shape (YAML + Markdown pair)

| Option | Fit here |
|--------|----------|
| **YAML + Markdown pair (chosen)** | YAML is structured for agents and tests; Markdown is the readable page. Matches existing repo docs (e.g. [alan-layer.md](../alan-layer.md)) without a new dependency beyond PyYAML (already in the environment) and Pydantic (already used). |
| STAC Item/Collection | Strong for cataloguing assets and extents. Weak for “what this predictor means / does not mean” narrative, and heavier than we need for an in-repo band catalog. Field names stay STAC-friendly where useful (`id`, licence, providers-as-source). |
| Raster attrs / xarray only | Good for machine association, poor as the only human doc. We still stamp attrs so docs resolve from a band. |

Keep the framework light: one YAML schema for layers, one for sources, one Markdown
page per documented band or upstream dataset, one catalog index. No separate
documentation site.

## Layout

```
docs/predictors/
  README.md           # this file
  SCHEMA.md           # field reference
  catalog.yml         # ordered lists of documented layer and source ids
  layers/
    <layer_id>.yml    # structured metadata + lineage
    <layer_id>.md     # human page (shows / does not show / caveats)
  sources/
    <source_id>.yml   # provider, product, licence, URLs, citation
    <source_id>.md    # human page for the upstream dataset
```

Layer ids match GeoTIFF band descriptions / `long_name` values after
`canonicalize_band_name` (see `build_model_ev_stack`).

Source ids are stable keys for upstream datasets. Layers point at them with
`source_id` instead of repeating licence and URL blocks on every band.

## Association (docs travel with the layer)

1. **Band name = layer id** — descriptions on `evs-to-model.tif` are the lookup key.
2. **Raster attrs** — optional stamps on each documented band:
   `sdm_layer_id`, `sdm_layer_doc`, `sdm_layer_kind`, `sdm_layer_family`
   (`sdm.predictors.attach_predictor_docs`).
3. **Sidecar** — `{raster}.predictors.yml` lists documented bands, doc paths, and
   `derived_from` lineage (`write_predictor_sidecar`). Safe to commit next to
   published stacks; contains no location data.

Undocumented bands are skipped until a later batch adds their YAML/Markdown.

## Lineage

Derived layers set `kind: derived` and `derived_from: [<parent_id>, …]`.
They reference parent docs; they do **not** copy parent caveats. Readers and
agents follow the parent id (or sidecar `derived_from`) for upstream limits.
Licence and access live on the parent’s `source_id` (or the source catalog entry
linked from that parent).

## Scope

- Document **only** layers already produced by the public pipeline on `main`.
- Do not add proposed or planned predictors here.
- Licence and access terms for upstream datasets live under `sources/`.

## Code entry points

```python
from sdm.predictors import load_catalog, load_layer, load_source, attach_predictor_docs

catalog = load_catalog()
doc = load_layer("terrain_dtm")
source = load_source(doc.source_id)  # e.g. ea_lidar_dtm
```

## Related docs

- Rebuild steps: [rebuild-ev-layers.md](../rebuild-ev-layers.md)
- Layer QA aggregates: [data_quality/environmental_layers.md](../data_quality/environmental_layers.md)
- ALAN (existing long-form page; catalog entry links the YAML): [alan-layer.md](../alan-layer.md)

## Implementation notes

**Modeller / data** — catalog covers every band listed in any
`data/sdm_config/*/variables_config.yml` on main, plus a few pipeline /
lineage parents (e.g. `terrain_dtm`, coast distance, ALAN, and
in-cell parents of multiscale land-cover bands). Follow the `derived_from`
chain for upstream caveats and `source_id` for licence / URL metadata.
