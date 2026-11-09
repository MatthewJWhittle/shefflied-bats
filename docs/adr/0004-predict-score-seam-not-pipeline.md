# 0004: Predict score seam must not require sklearn Pipeline

Date: 9 Nov 2026. Status: Accepted.

## Modelling

## Context
After the ModelAdapter registry landed, predict still loaded via the adapter
then scored with MaxEnt/sklearn assumptions: Pipeline `named_steps` peeking in
the predict command and `predict_proba(:, 1)` inside raster apply. DeepMaxent
(#111) currently fakes that shape with a Pipeline façade so predict stays
unchanged — a path adapters docs already reject long-term.

## Options
1. Deepen predict as “N Pipeline-like objects + `predict_proba[:, 1]`”.
   Rejected: cements the DeepMaxent façade.
2. Put EV raster scoring behind the predict adapter seam without requiring
   Pipeline or `[:, 1]` as the contract. Chosen.
3. Defer all predict deepening until DeepMaxent merges. Rejected: MaxEnt predict
   remains leaky; DeepMaxent still needs a non-façade seam later.

## Decision
- Predict asks the adapter to score models on an EV raster (one score call).
  Load, feature handling, and raster apply stay inside that path.
- The score seam’s contract does **not** require sklearn Pipeline or
  `predict_proba[:, 1]`. MaxEnt may still use those **inside** its adapter.
- The predict command must not inspect Pipeline internals. The unused feature-name
  peek is dropped; the scorer owns column selection. Non-Pipeline adapters may
  read `package.json` `feature_names` when they need an explicit list.
- The predict command still owns run resolution, EV path, outputs, and publish.
- After this deepen lands, a follow-up change should put DeepMaxent (#111) on
  the new seam and drop the Pipeline façade (separate PR; not this decision’s
  implementation).

## Why
So suitability scoring is testable through one interface, MaxEnt internals stay
local, and DeepMaxent can join the same seam without pretending to be a binary
sklearn Pipeline.

## Revisit when
- Joint multi-output scoring needs a different command-level call shape.
- Explainability still requires Pipeline inspection (separate from predict score).
