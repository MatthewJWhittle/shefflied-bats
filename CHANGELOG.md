# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Selectable CV fold methods (`kmeans`, `blocks`, `buffered_loo`) via `model.cv` / `sdm train --cv-method` ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)); see [docs/cv-folds.md](docs/cv-folds.md).
- Warning when `buffered_loo` is used with more presence records than `model.cv.buffered_loo_warn_above` / `--cv-buffered-loo-warn-above` (default 25); training is unchanged ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- Per-model `model_limited` flag (no published cut-off) and configurable pooled OOF percentile cut-off parameters under `model.threshold` ([#59](https://github.com/MatthewJWhittle/shefflied-bats/issues/59) / [#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- [docs/model-package-contract.md](docs/model-package-contract.md) — portable publish contract (toolchain → visualiser), aligned with [hsm-app](https://github.com/MatthewJWhittle/hsm-app) OpenAPI; study-area/species agnostic framing.

### Changed

- Default CV fold method is now 10 km spatial `blocks` (5 folds); `kmeans` and `buffered_loo` remain selectable ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- Published suitability cut-off rule is pooled held-out presence percentile (`pooled_oof_percentile`), replacing the per-fold 5th-percentile median from [#61](https://github.com/MatthewJWhittle/shefflied-bats/pull/61) ([#59](https://github.com/MatthewJWhittle/shefflied-bats/issues/59)).
- README modernised around the `sdm` CLI; legacy `data_prep/` workflow removed from docs.
