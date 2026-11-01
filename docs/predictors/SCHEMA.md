# Predictor layer YAML schema

Validated by `sdm.predictors.schema.LayerDoc` and `SourceDoc` (Pydantic).
Extra keys are rejected.

## Layer required

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Band id (snake case); must match the YAML filename stem. |
| `title` | string | Short human title. |
| `kind` | `source` \| `derived` | Source product vs processing-chain transform. |
| `family` | string | Grouping key: `terrain`, `terrain_stats`, `climate`, `landcover`, `vegetation`, `coastal`, `os`, `alan`, … |
| `summary` | string | One- or two-sentence plain description. |

## Layer recommended

| Field | Type | Meaning |
|-------|------|---------|
| `shows` | string[] | What the values represent. |
| `does_not_show` | string[] | Common misreadings / absences. |
| `units` | string | Physical or coded units. |
| `native_resolution_m` | number | Source grid before model resampling. |
| `model_resolution_m` | number | Model grid (default 100). |
| `crs` | string | Default `EPSG:27700`. |
| `temporal` | object | `label` plus optional `notes`. |
| `source_id` | string | Id of a doc under `sources/` (licence, URL, citation). Prefer this over repeating provider blocks on every layer. |
| `pipeline` | object | `rebuild_step` registry name; optional `code` path. |
| `derived_from` | string[] | Parent layer ids. **Required** when `kind: derived`; **forbidden** when `kind: source`. |
| `processing` | string[] | What this repo does to the source. |
| `caveats` | string[] | Limits for modelling and interpretation (ecology / data quality — not pipeline policy). |
| `references` | object[] | `{title, url}` public links. |
| `markdown` | string | Markdown filename under `layers/` (default `<id>.md`). |

Derived layers may omit `source_id` and inherit context via `derived_from`, or
set `source_id` when a direct pointer helps without opening the parent.

## Source docs (`sources/`)

One YAML + Markdown pair per upstream dataset. Validated by `SourceDoc`.

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Source id (snake case); must match the YAML filename stem. |
| `title` | string | Short human title. |
| `provider` | string | Organisation. |
| `product` | string | Dataset name. |
| `licence` | string | Short licence label (e.g. `OGL-UK-3.0`, `WorldClim-noncommercial`). |
| `licence_url` | string | Optional. |
| `access_url` | string | Optional download or service URL. |
| `citation` | string | Optional preferred citation. |
| `version` | string | Optional product version string. |
| `notes` | string | Optional licence or access caveats for the **dataset** (not repo implementation rules). |
| `references` | object[] | `{title, url}` public links. |
| `markdown` | string | Markdown filename under `sources/` (default `<id>.md`). |

## Catalog (`catalog.yml`)

```yaml
schema_version: 1
description: >
  Short blurb.
layers:
  - terrain_dtm
  - …
sources:
  - ea_lidar_dtm
  - …
```

Every layer id listed must have `layers/<id>.yml` and a Markdown page.
Every source id listed must have `sources/<id>.yml` and a Markdown page.
Layer `source_id` values must appear in `sources`.
