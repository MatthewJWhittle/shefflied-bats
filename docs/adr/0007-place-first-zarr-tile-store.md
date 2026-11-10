# 0007: Place-first rebuild; Zarr national cache; COG publish

Date: 10 Nov 2026. Status: Accepted.

## Data

## Context
ADR 0002 fixed the internal geography unit (10 240 m national tiles, EPSG:27700,
halo, rebuild only touched tiles). ADR 0006 fixed rebuild **what** (layered
pieces; merge ⇒ stack; `--only stack` from existing `all-evs`). Operators still
faced a UK-scale memory story that wants tiles, while everyday work is “rebuild
this study area.” Issue #41 needs a versioned per-layer store with cache /
invalidate, and published maps still go through TiTiler / COG packages.

## Options
1. **Option A** — tile key and AOI as equal everyday knobs. Rejected: operators
   think in places; tiles are how work scales.
2. **Option B** — place-first; tiles mostly hidden; rare tile-key escape hatch.
   Chosen.
3. **Option C** — pieces-only with where always implicit from the current folder.
   Rejected: no clear polygon entry for a new site or partial national refresh.
4. National intermediate store as tiled GeoTIFF/COG first, Zarr later. Rejected
   for the intermediate system of record: adopt Zarr in the design now.
5. Live XYZ / TiTiler from national Zarr. Rejected as the primary map path;
   COG remains the publish / TiTiler surface (`titiler.xarray` optional later).

## Decision
- **Where (operator):** everyday extent is a **study-area polygon** (AOI). The
  system resolves intersecting ADR 0002 tiles (+ halo implied by the pieces).
  **Rare escape hatch:** rebuild one national **tile key** (failed square, QA,
  local source fix). One request uses one where mode — polygon *or* tile key,
  not both conflicting extents.
- **No place ids:** where input is a polygon (or rare tile key) only. No named
  sites, cell indexes, or discrete place identifiers in the data model for now.
- **Tile key:** deterministic, simple, scalable (exact encoding — e.g. `col_row`
  vs southwest metres — chosen at implementation; same key always means the same
  national square). The key names the **interior** tile, not the halo.
- **Durable compute / cache store:** **Zarr**, one array (or group) **per
  environmental feature**, chunked to the ADR 0002 national tile grid.
- **Published delivery:** **COG** for model packages, TiTiler, and the
  visualiser. Assembly (window / VRT / Dask mosaic over selected chunks) produces
  AOI products such as `all-evs` / `evs-to-model`; those remain named products.
  Prefer treating AOI mosaics as **derived** from the per-feature tile store.
- **Cache identity:** a stored unit is `(feature_id, tile_key, inputs_hash)`.
  Validity is that identity (and bytes present), not wall-clock age alone.
  Timestamps are for observability / optional freshness *gates*, not “still
  correct.”
- **Aligns with #41:** versioned layers, model→layer-version pinning, no full
  duplicate of unchanged layers. Exact manifest home, hash payload depth, and
  run-folder vs national Zarr layout remain open design questions (tracked in
  GitHub issues under #132 (especially #133 for cache semantics; #134 for remaining grill)), not settled here.
- **Out of scope for this record:** implementing the Zarr store; Oct 2025 /
  BLE release modelling work (separate PR stream); finishing architecture grill
  candidates C5–C8; DBT-style selective `--select` (still deferred from ADR 0006).

## Why
So UK-scale rebuild keeps peak memory flat (ADR 0002) without forcing operators
to name tiles every day; the national store matches “one format per job”
(Zarr for chunked compute/cache, COG for publish/serve); and cache hits follow
content identity that #41 and released models can pin.

## Now / Instead

| | Now (study-area rebuild) | Instead (this decision) |
|--|--------------------------|-------------------------|
| Extent | Whole study boundary every time | Primary: study-area polygon → tiles. Rare: one tile key |
| Durable store | Whole-AOI mosaics / folders | Per-feature Zarr on the ADR 0002 grid; COG on publish |
| Cache | Limited skip/re-entry by rebuild step | `(feature, tile_key, inputs_hash)` hit/miss (semantics still to design) |
| Map serve | COG / TiTiler path | Unchanged: COG for TiTiler; not live XYZ from national Zarr |

## Revisit when
- Manifest / `inputs_hash` / concurrency / halo-in-key / freshness design closes
  enough to implement #41 against this store shape.
- Published delivery needs a different primary format than COG.
- Place ids or multi-extent requests become a real product need.
