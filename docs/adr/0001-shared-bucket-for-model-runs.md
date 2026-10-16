# 0001: Shared bucket so models train the same on either machine

Date: 16 Oct 2026. Status: accepted (Matt, hsm room).

## Context
The goal is to train models on Matt's laptop or Andrej's computer from the same config and inputs and get broadly the same result. Both machines need one shared place for inputs and run outputs.

## Options
1. Tightly scoped permissions per folder (read inputs/, create-only runs/). Tried in #25. It kept blocking normal work and needed more rules each time.
2. One identity per machine with broad object access to a single bucket. Chosen.
3. Cloud compute now (for example Vertex AI). Deferred.

## Decision
- Bucket `habitat-suitability-modelling-data`. Inputs live under `inputs/` and each run writes to `runs/<UTC date-time>-<id>/`. The full list is in `docs/data-layout.md`. New top-level folders only when needed.
- Matt uses his own Google login. Remote machines (Andrej's for now) use the `model-run` service account through `GOOGLE_APPLICATION_CREDENTIALS`, with a key file kept only on that machine and never in git.
- `model-run` has `roles/storage.objectAdmin` on the bucket: read, write, overwrite. It can't change bucket settings or access. Overwriting counts as deleting the old version in Cloud Storage, so it can delete single files. The 30-day version history covers that.
- Code reads and writes through plain fsspec paths under `MODEL_OUTPUT_ROOT`, with no storage-specific commands in the command-line tool.

## Why
This is the simplest setup that meets the goal (Matt: keep it simple, no tight control). Versioning makes broad access safe enough for a two-person project.

## How it's judged
One training run on each machine with the same config and inputs gives broadly the same results, and both runs are readable from the bucket.

## Revisit when
- Cloud compute is added.
- Anyone beyond Matt and Andrej needs access.
- Real (non-public) data goes in the bucket.
- The bucket's monthly cost goes over the £1 budget alert.

Known gaps: tuning still saves its chosen settings locally, and prediction and explanation read local files only.
