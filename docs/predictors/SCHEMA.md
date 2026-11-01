# Predictor layer YAML schema

Validated by `sdm.predictors.schema.LayerDoc` (Pydantic). Extra keys are rejected.

## Required

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Band id (snake case); must match the YAML filename stem. |
| `title` | string | Short human title. |
| `kind` | `source` \| `derived` | Source product vs processing-chain transform. |
| `family` | string | Grouping key: `terrain`, `terrain_stats`, `climate`, `landcover`, `vegetation`, `coastal`, `os`, `alan`, … |
| `summary` | string | One- or two-sentence plain description. |

## Recommended

| Field | Type | Meaning |
|-------|------|---------|
| `selection` | `included` \| `excluded` \| `available` | `included`: used in the model stack (CSV `include=1`, or a multiscale band grown from an included parent). `excluded`: deliberately omitted from the model stack (CSV `include=0`). `available`: rebuild may write it, but it is not a current model feature. Default `included`. |
| `shows` | string[] | What the values represent. |
| `does_not_show` | string[] | Common misreadings / absences. |
| `units` | string | Physical or coded units. |
| `native_resolution_m` | number | Source grid before model resampling. |
| `model_resolution_m` | number | Model grid (default 100). |
| `crs` | string | Default `EPSG:27700`. |
| `temporal` | object | `label` plus optional `notes`. |
| `source` | object | Public provider / product / licence (see below). |
| `pipeline` | object | `rebuild_step` registry name; optional `code` path. |
| `derived_from` | string[] | Parent layer ids. **Required** when `kind: derived`; **forbidden** when `kind: source`. |
| `processing` | string[] | What this repo does to the source. |
| `caveats` | string[] | Limits for modelling and interpretation. |
| `references` | object[] | `{title, url}` public links. |
| `markdown` | string | Markdown filename under `layers/` (default `<id>.md`). |

### `source` object

| Field | Meaning |
|-------|---------|
| `provider` | Organisation. |
| `product` | Dataset name. |
| `licence` | Short licence label (e.g. `OGL-UK-3.0`, `WorldClim-noncommercial`). |
| `licence_url` | Optional. |
| `access_url` | Optional download or service URL. |
| `citation` | Optional preferred citation. |
| `version` | Optional product version string. |
| `notes` | Optional licence or access caveats (e.g. CEH non-redistribution). |

Derived layers may omit `source` and inherit context via `derived_from`, or
repeat a short `source` pointer when that helps agents without opening the parent.

## Catalog (`catalog.yml`)

```yaml
schema_version: 1
description: >
  Short blurb.
layers:
  - terrain_dtm
  - …
```

Every id listed must have `layers/<id>.yml` and a Markdown page.
