# 0006: Environmental layer rebuild has one locality

Date: 9 Nov 2026. Status: Accepted.

## Data

## Context
`sdm data` rebuilds environmental layers, but step identity, dry-run planning,
provenance URLs, and the real runner lived in separate modules. CLI step names
and recorder keys differed; CEH / WorldClim / EA WCS constants were duplicated.
`build_model_ev_stack` always followed merge but was not a public `--skip` /
`--only` name. Rebuild is heavy; Matt wants layered re-entry (whole stack or
pieces) and later reuse for a different layer set.

## Options
1. Leave plan / registry / runner / provenance as hand-aligned siblings.
2. Deepen one rebuild locality: one public step id, source constants once, thin
   CLI; layer steps as composable pieces; merge implies stack by default with
   `--only stack` on existing `all-evs.tif`. Chosen.
3. Build DBT-style selective caching / `--select` now. Deferred: future work,
   not this deepen.

## Decision
- Plan, run, and provenance share one rebuild locality so step identity and
  source constants cannot drift.
- One public rebuild-step id for `--skip` / `--only` / plan / metrics; longer
  function names stay private.
- Source constants (CEH path, WorldClim URL pattern, EA WCS endpoints) live once.
- CLI stays a thin caller of that locality.
- Layer generate steps are composable pieces: rebuild the whole set, one piece,
  or later a different stack (drop/add layers) with the same composition code.
- Merge (`all-evs.tif`) implies model stack (`evs-to-model.tif`) by default.
  Stack is also independently runnable (e.g. `--only stack`) and then reads the
  existing `all-evs.tif` without re-merging.
- Selective rebuild / caching like DBT `--select` is explicitly **out of scope**
  for this deepen; re-entry without recomputing everything is the design aim
  that later caching should serve.

## Why
So adding or swapping layers does not require hand-syncing four modules, merge
vs stack behaviour matches how people iterate (full merge path vs feature-only
restack), and UK-scale tiling work (ADR 0002) can plug into clear rebuild-step
pieces later.

## Revisit when
- DBT-style selective rebuild / artefact caching is designed.
- A different product than `evs-to-model.tif` becomes a first-class stack output.
