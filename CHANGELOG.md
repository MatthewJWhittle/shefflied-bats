# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Opt-in versioned prediction publishing (`sdm predict --publish`): writes `prediction__<genus>_<species>_<behaviour>__<version>__suitability.tif`, `…__threshold_<rule>.tif`, and `…__run.json` under `predictions/<model_id>/<YYYYMMDD>_<sha7>_cfg<hash>/` without overwriting existing versions.
- Shared MLflow run tracking helpers: Neon Postgres backend (no tracking server), GCS artefact prefix, Secret Manager / `MLFLOW_DB_PASSWORD` for the tracking DB password, example settings file, and dual-writer tests ([#93](https://github.com/MatthewJWhittle/shefflied-bats/issues/93)); see [docs/mlflow-tracking.md](docs/mlflow-tracking.md).
- Selectable CV fold methods (`kmeans`, `blocks`, `buffered_loo`) via `model.cv` / `sdm train --cv-method` ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)); see [docs/cv-folds.md](docs/cv-folds.md).
- Warning when `buffered_loo` is used with more presence records than `model.cv.buffered_loo_warn_above` / `--cv-buffered-loo-warn-above` (default 25); training is unchanged ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- Per-model `model_limited` flag so data-limited models publish no suitability cut-off ([#67](https://github.com/MatthewJWhittle/shefflied-bats/issues/67)); whiskered in flight (`Myotis_mystacinus_In_flight`) is flagged.
- [docs/model-package-contract.md](docs/model-package-contract.md) — portable publish contract (toolchain → visualiser), aligned with [hsm-app](https://github.com/MatthewJWhittle/hsm-app) OpenAPI; study-area/species agnostic framing.

### Changed

- Prediction publish root moved out of library defaults into `prediction_publish.root` in `config.yml` (`--publish-root` / `PREDICTION_PUBLISH_ROOT` still win); `--publish` with no root fails early.
- Prediction publish file names are built from each model's full Latin binomial (snake case) + behaviour; the short-name → Latin species map is removed. Publishing fails clearly when `latin_name` is missing.
- `sdm setup` writes the modelling boundary as counties buffered by `study_area_buffer` then clipped to ONS BFC land, so EV builders no longer expand into the sea / estuary ([#78](https://github.com/MatthewJWhittle/shefflied-bats/issues/78)).
- Cap vegetation height at 40 m in the environmental-variables layer path: raw VOM before summary metrics, and mean/min/max bands in model-stack layer clipping ([#73](https://github.com/MatthewJWhittle/shefflied-bats/issues/73)).
- Default CV fold method is now 10 km spatial `blocks` (5 folds); `kmeans` and `buffered_loo` remain selectable ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- Published suitability cut-off rule is pooled held-out presence 10th percentile (`pooled_oof_percentile`) for every model, replacing the per-fold 5th-percentile median from [#61](https://github.com/MatthewJWhittle/shefflied-bats/pull/61) ([#67](https://github.com/MatthewJWhittle/shefflied-bats/issues/67)).
- README modernised around the `sdm` CLI; legacy `data_prep/` workflow removed from docs.
