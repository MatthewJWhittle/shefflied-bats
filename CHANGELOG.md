# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Selectable CV fold methods (`kmeans`, `blocks`, `buffered_loo`) via `model.cv` / `sdm train --cv-method` ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)); see [docs/cv-folds.md](docs/cv-folds.md).
- Warning when `buffered_loo` is used with more presence records than `model.cv.buffered_loo_warn_above` / `--cv-buffered-loo-warn-above` (default 25); training is unchanged ([#63](https://github.com/MatthewJWhittle/shefflied-bats/issues/63)).
- [docs/model-package-contract.md](docs/model-package-contract.md) — portable publish contract (toolchain → visualiser), aligned with [hsm-app](https://github.com/MatthewJWhittle/hsm-app) OpenAPI; study-area/species agnostic framing.

### Changed

- README modernised around the `sdm` CLI; legacy `data_prep/` workflow removed from docs.
